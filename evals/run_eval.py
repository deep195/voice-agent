"""Run the golden set and score the agent.

    .venv\\Scripts\\python.exe evals\\run_eval.py              # all cases
    .venv\\Scripts\\python.exe evals\\run_eval.py --tags knowledge
    .venv\\Scripts\\python.exe evals\\run_eval.py --id capital_france -v
    .venv\\Scripts\\python.exe evals\\run_eval.py --baseline last.json

Design notes
------------
The golden set is behavioural, not exact-match. A 3B model rewording a correct answer is
not a regression, so assertions are "contains a fact", "never says X", "used the right
tool", "stayed short". Latency and speech round-trip are scored too, because a voice agent
that is accurate but takes 20s is broken in use.

Results append to evals/results.jsonl so runs accumulate into a trend rather than being
overwritten.
"""

import argparse
import json
import re
import statistics
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import config, llm, stt, tts, tools  # noqa: E402
from agent.agent import Agent  # noqa: E402
from agent.tts import for_speech  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden.jsonl"
RESULTS = Path(__file__).resolve().parent / "results.jsonl"

# Latency budgets in seconds. Generous: this is a 4-core laptop CPU, not a datacentre.
# A tool-using turn costs an extra model round trip plus the network call, so those are
# budgeted separately rather than by loosening llm_first for everything.
BUDGET = {
    "stt": 4.0,
    "llm_first": 25.0,
    "turn": 45.0,
    "web_first": 60.0,
    "web_turn": 90.0,
}

# "How can I assist you today?" every single turn is the most audible LLM tic in a voice
# interface, so it is an assertion rather than something left to taste.
SIGNOFF = re.compile(
    r"(how (?:can|may) i (?:assist|help)[^.?]*[.?]?"
    r"|is there anything else[^.?]*[.?]?"
    r"|let me know if you (?:need|have)[^.?]*[.?]?"
    r"|feel free to[^.?]*[.?]?"
    r"|don't hesitate to[^.?]*[.?]?)",
    re.IGNORECASE,
)


def load_cases(path: Path, tags=None, only=None) -> list[dict]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        case = json.loads(line)
        if only and case["id"] not in only:
            continue
        if tags and not set(tags) & set(case.get("tags", [])):
            continue
        cases.append(case)
    return cases


def normalize(text: str) -> str:
    """Lowercase, strip accents and punctuation, so 'Paris.' matches 'paris'."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(
        "".join(c if c.isalnum() or c in " -" else " " for c in text.lower()).split()
    )


class Judge:
    """Wraps one agent so a case can be run as a full voice turn."""

    def __init__(self, model: str, listen, speak: bool, verbose: bool) -> None:
        self.model = model
        self.listen = listen
        self.speak = speak
        self.verbose = verbose
        self.speaker = None
        self.tools_used: list[str] = []
        self.original_run = tools.run

    def __enter__(self):
        if self.speak:
            self.speaker = tts.Speaker(config.TTS_BACKEND, config.TTS_RATE, "", config.NEURAL_LENGTH_SCALE)
            self.speaker.start()

        def spy(name, arguments):
            self.tools_used.append(name)
            return self.original_run(name, arguments)

        tools.run = spy
        return self

    def __exit__(self, *exc):
        tools.run = self.original_run
        if self.speaker is not None:
            self.speaker.shutdown()

    def ask(self, agent: Agent, utterance: str) -> dict:
        """Run one turn. Returns reply text, transcript and timings."""
        result = {"transcript": utterance, "tools": [], "llm_first": 0.0, "turn": 0.0, "stt": 0.0}
        started = time.time()
        text_parts: list[str] = []
        first = None
        for piece in agent.turn(utterance):
            text_parts.append(piece)
            if first is None:
                first = time.time() - started
        result["llm_first"] = first or 0.0
        result["turn"] = time.time() - started
        result["reply"] = "".join(text_parts).strip()
        result["tools"] = list(self.tools_used)
        self.tools_used.clear()

        if self.listen is not None and utterance:
            # Score what the ears actually hear, not the text we fed in.
            result["stt"] = self._transcribe_roundtrip(utterance, result["reply"])
        return result

    def _transcribe_roundtrip(self, utterance: str, reply: str) -> float:
        """Synthesize the question and transcribe it back, timing the ears."""
        try:
            wav = self.speaker.preview(utterance)
            audio, rate = wav[0], wav[1]
            import numpy as np

            if rate != config.SAMPLE_RATE:
                count = int(len(audio) * config.SAMPLE_RATE / rate)
                audio = np.interp(
                    np.linspace(0, len(audio) - 1, count),
                    np.arange(len(audio)),
                    audio,
                ).astype(np.float32)
            started = time.time()
            heard = self.listen.transcribe(audio)
            elapsed = time.time() - started
            # Stored on the listener: main() reads listener._last_heard for the "heard"
            # column of each row, and Judge.last_heard was never read by anything.
            self.listen._last_heard = heard
            expected = normalize(utterance).split()
            got = normalize(heard).split()
            # Word error rate between what we said and what the ears recovered.
            if expected:
                distance = _levenshtein(expected, got)
                setattr(self.listen, "_wer", distance / len(expected))
            return elapsed
        except Exception:
            return 0.0


def _as_list(value) -> list:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def _levenshtein(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    previous = list(range(len(b) + 1))
    for i, word_a in enumerate(a, start=1):
        current = [i]
        for j, word_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (word_a != word_b),
                )
            )
        previous = current
    return previous[-1]


def score(case: dict, result: dict) -> tuple[bool, list[str]]:
    """Return (passed, list of failure reasons).

    Scored on what the voice would actually say: for_speech() is applied first, so markdown
    that the synthesizer silently strips anyway is not counted against the reply.
    """
    problems: list[str] = []
    reply = for_speech(result.get("reply", ""), final=True)
    flat = normalize(reply)
    expect = case.get("expect", {})

    if case.get("silent"):
        return True, problems

    if not reply:
        problems.append("empty reply")
        return False, problems

    if expect.get("contains_any") and not any(
        normalize(term) in flat for term in expect["contains_any"]
    ):
        problems.append(f"missing any of {expect['contains_any']!r}")
    if expect.get("contains_all") and not all(
        normalize(term) in flat for term in expect["contains_all"]
    ):
        problems.append(f"missing all of {expect['contains_all']!r}")
    if expect.get("regex"):
        import re

        if not re.search(expect["regex"], reply, re.IGNORECASE):
            problems.append(f"regex {expect['regex']!r} did not match")
    if expect.get("forbid"):
        # Checked against the spoken text. normalize() would turn "**" into "" and match
        # everything, and tts.normalize() already rewrites the bare forms of these.
        spoken = reply.lower()
        for term in expect["forbid"]:
            if term.lower() in spoken:
                problems.append(f"said forbidden {term!r}")

    if expect.get("tool") and expect["tool"] not in result.get("tools", []):
        problems.append(f"did not call {expect['tool']}")
    allowed = case.get("allow_tool")
    if allowed:
        used = set(result.get("tools", []))
        if not used & set(allowed):
            problems.append(f"used none of {allowed!r} (used {sorted(used)})")

    limit = case.get("max_reply_words")
    if limit and len(reply.split()) > limit:
        problems.append(f"too long: {len(reply.split())} words > {limit}")

    # Scored on the spoken text: agent.tts.strip_signoffs removes the canned tail before
    # synthesis, so this asserts what the user actually hears rather than what the model
    # typed. agent/tts.py has its own unit coverage for the stripper itself.
    if not case.get("allow_signoff") and SIGNOFF.search(reply):
        problems.append(f"audibly ends with a canned sign-off: {SIGNOFF.search(reply).group(0)!r}")

    used_web = bool(set(result.get("tools", [])) & {"web_search", "web_fetch"})
    if result.get("stt", 0) > BUDGET["stt"]:
        problems.append(f"stt too slow: {result['stt']:.1f}s")
    first_budget = BUDGET["web_first"] if used_web else BUDGET["llm_first"]
    turn_budget = case.get("max_turn_seconds") or (BUDGET["web_turn"] if used_web else BUDGET["turn"])
    if result.get("llm_first", 0) > first_budget:
        problems.append(f"first token too slow: {result['llm_first']:.1f}s")
    if result.get("turn", 0) > turn_budget:
        problems.append(f"turn too slow: {result['turn']:.1f}s")

    wer = getattr(stt_active[0], "_wer", None) if stt_active else None
    if wer is not None and wer > 0.35:
        problems.append(f"asr word error rate {wer:.0%} on the question")

    return not problems, problems


stt_active = []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=config.LLM_MODEL)
    parser.add_argument("--tags", nargs="*")
    parser.add_argument("--id", nargs="*")
    parser.add_argument("--listen", action="store_true", help="also score the speech path")
    parser.add_argument("--no-speak", action="store_true", help="skip TTS entirely (fastest)")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--baseline", help="compare against a previous results file")
    parser.add_argument("--out")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 by default so runs are comparable; the app itself uses 0.6",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if not llm.is_available():
        print(f"ollama is not answering at {config.OLLAMA_HOST}")
        return 2

    # Greedy decoding by default: at the app's 0.6 two runs of the same suite disagree on
    # several cases, which makes a score impossible to compare against a baseline.
    config.LLM_TEMPERATURE = args.temperature

    ALL_CASES = {case["id"]: case for case in load_cases(GOLDEN)}
    cases = load_cases(GOLDEN, args.tags, args.id)
    if not cases:
        print("no cases matched")
        return 2

    listener = None
    if args.listen and not args.no_speak:
        listener = stt.create(config.STT_BACKEND)
        listener.load()
        stt_active.append(listener)

    speak = not args.no_speak
    print(
        f"model {args.model} | web {'on' if config.WEB_ENABLED else 'off'} | "
        f"{len(cases)} cases x {args.repeat} | "
        f"{'speak+listen' if listener else ('speak' if speak else 'text only')}\n"
    )

    by_id = {case["id"]: case for case in cases}
    # A case may depend on a setup case that --id/--tags filtered out, so resolve those
    # from the full set or a single-case run dies with a KeyError.
    for case in cases:
        for setup_id in _as_list(case.get("after")):
            by_id.setdefault(setup_id, ALL_CASES.get(setup_id))
    if any(case is None for case in by_id.values()):
        missing = sorted(k for k, v in by_id.items() if v is None)
        print(f"unknown setup case(s): {', '.join(missing)}")
        return 2
    rows: list[dict] = []
    passed_count = 0
    total = 0

    with Judge(args.model, listener, speak, args.verbose) as judge:
        for attempt in range(args.repeat):
            agent = Agent(args.model)
            for case in cases:
                if case.get("after"):
                    # `after` may be a single id or a list; json gives a bare string
                    # when there is only one, and iterating that yields letters.
                    for setup_id in _as_list(case["after"]):
                        judge.ask(agent, by_id[setup_id]["utterance"])
                total += 1
                started = time.time()
                result = judge.ask(agent, case["utterance"])
                passed, problems = score(case, result)
                passed_count += passed
                row = {
                    "id": case["id"],
                    "tags": case.get("tags", []),
                    "model": args.model,
                    "attempt": attempt,
                    "passed": passed,
                    "problems": problems,
                    "reply": result["reply"],
                    "heard": getattr(listener, "_last_heard", "") if listener else "",
                    "tools": result["tools"],
                    "stt": round(result.get("stt", 0), 2),
                    "llm_first": round(result.get("llm_first", 0), 2),
                    "turn": round(result["turn"], 2),
                    "wall": round(time.time() - started, 2),
                }
                rows.append(row)
                mark = "PASS" if passed else "FAIL"
                print(
                    f"  {mark} {case['id']:<24} "
                    f"{row['turn']:>5.1f}s llm {row['llm_first']:>4.1f}s "
                    f"tools {','.join(row['tools']) or '-':<18} {result['reply'][:56]}"
                )
                for problem in problems:
                    print(f"       - {problem}")

    print(f"\n{passed_count}/{total} passed ({passed_count / total:.0%})")
    turns = [row["turn"] for row in rows]
    if turns:
        print(
            f"turn latency: median {statistics.median(turns):.1f}s "
            f"p90 {sorted(turns)[int(len(turns) * 0.9) - 1]:.1f}s max {max(turns):.1f}s"
        )

    # Resolve before writing: a bare filename like "out.jsonl" is a relative path, and
    # relative_to() below raises on anything that is not already under the project root.
    out = (Path(args.out) if args.out else RESULTS).resolve()
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    with out.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"at": stamp, **row}) + "\n")
    try:
        shown = out.relative_to(ROOT)
    except ValueError:
        shown = out
    print(f"appended {len(rows)} rows to {shown}")

    if args.baseline:
        compare(Path(args.baseline), rows)
    return 0 if passed_count == total else 1


def compare(baseline: Path, rows: list[dict]) -> None:
    """Diff this run against a stored results file."""
    before: dict[str, list[bool]] = {}
    for line in baseline.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            before.setdefault(record["id"], []).append(record["passed"])

    def rate(flags: list[bool]) -> float:
        return sum(flags) / len(flags) if flags else 0.0

    now = {row["id"]: [row["passed"]] for row in rows}
    print(f"\nvs {baseline.name}:")
    regressed = []
    improved = []
    for case_id in sorted(set(before) | set(now)):
        was = rate(before.get(case_id, []))
        is_now = rate(now.get(case_id, []))
        if is_now < was:
            regressed.append(case_id)
        elif is_now > was:
            improved.append(case_id)
    print(f"  improved: {improved or 'none'}")
    print(f"  regressed: {regressed or 'none'}")
    was_total = rate([flag for flags in before.values() for flag in flags])
    now_total = rate([row["passed"] for row in rows])
    print(f"  overall: {was_total:.0%} -> {now_total:.0%}")


if __name__ == "__main__":
    raise SystemExit(main())