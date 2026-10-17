"""Measure the voice, not the words.

The question this answers is "does it sound like a person?", which run_eval.py cannot answer
because that suite only ever looks at text. Everything here is measured off the rendered
audio, so it catches the failures a transcript hides: an intonation seam where two chunks
join, a chunk that arrives too late to keep up with playback, a reply that trails off, a
waveform pinned at full scale.

Metrics, and why each one is here:

    rate      words per minute of audio. Natural English conversation sits near 150-170.
              Below ~120 sounds slow or hesitant, above ~185 sounds rushed.
    seam      pitch discontinuity at every chunk join, as a fraction of the pitch. This is
              the robotic-sounding artefact: each chunk is synthesised independently, so
              the pitch can land somewhere unrelated to where the previous chunk ended.
    catchup   synthesis time divided by audio duration (the realtime factor). Below 1.0 the
              synth thread keeps ahead of the speaker; above 1.0 the voice runs dry
              mid-reply no matter how good the audio is.
    lead      milliseconds from starting synthesis to the first playable chunk. This is
              what the user actually waits for after the model stops talking.
    trail     silence appended to the end, i.e. the dead air before the agent feels done.
    headroom  peak sample level. Anything at 1.0 is pinned at full scale and distorts.

Nothing here is a vibe check pretending to be a measurement: every number is computed from
the samples, and the thresholds are the ones stated above rather than fitted to pass.

    .venv\\Scripts\\python.exe evals\\voice_eval.py
    .venv\\Scripts\\python.exe evals\\voice_eval.py --json out.json --repeat 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import config  # noqa: E402
from agent.tts import Speaker, for_speech, stream_chunks  # noqa: E402


# A reply is not one sentence: the numbers, the abbreviation and the long clause are what
# break chunking and normalisation, so the corpus has to contain them.
CORPUS = [
    "The capital of France is Paris.",
    "Sure, I can help with that. The population of Bristol is around four hundred and ninety "
    "five thousand, according to the most recent figures. Would you like me to look up "
    "anything else for you today?",
    "Water freezes at 0 degrees Celsius and boils at 100 degrees, so the range is narrow.",
    "Dr. Alvarez said the etc. was fine, roughly 25 percent of the total, which is about "
    "$1,250 per month at 3.5 percent interest.",
    "I could not find a direct answer to that. What time frame did you have in mind?",
    "The three steps are: download the installer, run it, and make sure you tick the box "
    "that adds Python to your PATH variable.",
]

# Voice-agent replies that should sound unhurried rather than read.
TARGETS = {
    "rate": (150, 175),      # wpm
    "seam": (0.0, 0.12),     # fractional pitch jump at a chunk join
    "catchup": (0.0, 1.0),   # realtime factor; must keep ahead of playback
    "trail": (0.0, 0.12),    # trailing silence, seconds
    "headroom": (0.0, 0.98), # peak level; 1.0 is distortion
}

PRESETS = [
    # name,               voice,          speed, min_words, debounce, gap
    ("kokoro default",    "af_bella",     1.05,  4,          0.28,     0.05),
    ("kokoro conversational", "af_bella", 1.20,  5,          0.35,     0.08),
    ("kokoro slow",       "af_bella",     1.25,  6,          0.40,     0.10),
    # Chunk size turns out to dominate everything else here: synthesis carries a fixed cost
    # per call, so a four-word chunk spends most of its time on overhead rather than audio.
    # These four isolate that, holding voice and speed still.
    ("chunks 8w",         "af_bella",     1.20,  8,          0.35,     0.08),
    ("chunks 12w",        "af_bella",     1.20,  12,         0.35,     0.08),
    ("chunks 18w",        "af_bella",     1.20,  18,         0.35,     0.08),
    ("chunks 24w",        "af_bella",     1.20,  24,         0.35,     0.08),
    ("chunks 18w heart",  "af_heart",     1.20,  18,         0.35,     0.08),
    ("chunks 18w bf_emma", "bf_emma",     1.20,  18,         0.35,     0.08),
]


# ---------- signal analysis ----------

F0_MIN, F0_MAX = 70.0, 400.0   # Hz; 400 is above a soprano's top


def frame_f0(x: np.ndarray, rate: int) -> float:
    """Median fundamental frequency of a voiced stretch, by autocorrelation.

    Autocorrelation rather than a spectral peak because it does not care about the harmonic
    structure, and this only needs to be accurate enough to compare one chunk with its
    neighbour. Returns NaN when there is no measurable pitch, rather than guessing.
    """
    # A window has to span several pitch periods to resolve them: the slowest voice we care
    # about is 70 Hz (14 ms per cycle), so 85 ms covers it several times over. Shorter input
    # is used as-is rather than padded, since silence either side would only blur the peak.
    frame = min(int(rate * 0.085), len(x))
    if frame < 512:
        return float("nan")
    # Autocorrelation lags are measured in samples, so the search range is rate/frequency.
    lo = max(1, int(rate / F0_MAX))
    hi = int(rate / F0_MIN)
    if hi >= frame:
        return float("nan")
    seg = x[:frame] if len(x) <= frame * 2 else x[len(x) // 2 - frame // 2 :][:frame]
    seg = seg - seg.mean()
    corr = np.correlate(seg, seg, mode="full")[frame - 1 :]
    if corr[0] <= 0:
        return float("nan")
    window = corr[lo:hi]
    if not len(window):
        return float("nan")
    peak = int(np.argmax(window)) + lo
    # Reject noise: a real pitch shows a strong peak, not a flat correlation.
    if corr[peak] <= 0.3 * corr[0]:
        return float("nan")
    f0 = rate / peak
    return f0 if F0_MIN <= f0 <= F0_MAX else float("nan")


def trim_edges(x: np.ndarray, thresh: float = 0.01):
    """Strip leading and trailing silence, keeping everything in between.

    Synthesised chunks start and end with near-silence, so the literal first and last
    milliseconds carry no pitch at all. Inner pauses are deliberately kept: the gaps between
    words are part of how long the sentence takes to say.
    """
    loud = np.abs(x) >= thresh
    if not loud.any():
        return None
    first = int(loud.argmax())
    last = len(loud) - 1 - int(loud[::-1].argmax())
    return x[first : last + 1]


def voiced_seconds(x: np.ndarray, rate: int, thresh: float = 0.01) -> float:
    """Seconds from the first voiced sample to the last, ignoring outer padding only."""
    trimmed = trim_edges(x, thresh)
    return 0.0 if trimmed is None else len(trimmed) / rate


def seam_jump(prev: np.ndarray, nxt: np.ndarray, rate: int) -> float:
    """Pitch discontinuity where two chunks meet, as a fraction of pitch.

    Compares the pitch at the tail of the previous chunk with the pitch at the head of the
    next one. Natural speech glides between them; two independent syntheses usually do not.
    Returns NaN when either side has no measurable pitch, which the caller must treat as a
    failed measurement rather than a good score.
    """
    prev_voice, next_voice = trim_edges(prev), trim_edges(nxt)
    if prev_voice is None or next_voice is None:
        return float("nan")
    a = frame_f0(prev_voice[-int(rate * 0.15) :], rate)
    b = frame_f0(next_voice[: int(rate * 0.15)], rate)
    if a != a or b != b or not a or not b:
        return float("nan")
    return abs(b - a) / a


def trailing_silence(x: np.ndarray, rate: int, thresh: float = 0.01) -> float:
    """Seconds of near-silence at the end of the audio."""
    if not len(x):
        return 0.0
    quiet = np.abs(x) < thresh
    idx = len(quiet) - 1
    while idx >= 0 and quiet[idx]:
        idx -= 1
    return (len(quiet) - 1 - idx) / rate


# ---------- measurement ----------

def measure(preset, repeat: int) -> dict:
    name, voice, speed, min_words, debounce, gap = preset
    config.NEURAL_MIN_CHUNK_WORDS = min_words
    config.NEURAL_FINAL_DEBOUNCE = debounce
    config.NEURAL_GAP_SECONDS = gap

    speaker = Speaker(config.TTS_BACKEND, config.TTS_RATE, voice, speed)
    speaker.reconfigure(voice=voice, speed=speed)
    speaker.preview("warm the engine")      # excluded from every timing below

    rows = []
    for _ in range(repeat):
        for text in CORPUS:
            chunks = list(stream_chunks([text], min_words=min_words))
            audio_parts, seams = [], []
            synth_start = time.perf_counter()
            lead = None
            for chunk in chunks:
                spoken = for_speech(chunk, final=False)
                if not spoken:
                    continue
                wav, rate = speaker.preview(spoken)
                if lead is None:
                    lead = (time.perf_counter() - synth_start) * 1000
                if audio_parts:
                    seams.append(seam_jump(audio_parts[-1], wav, rate))
                audio_parts.append(wav)
            synth_ms = (time.perf_counter() - synth_start) * 1000
            if not audio_parts:
                continue
            audio = np.concatenate(audio_parts)
            dur = len(audio) / rate
            spoken_s = voiced_seconds(audio, rate)
            words = len(text.split())
            measured = [s for s in seams if s == s]
            rows.append({
                # WPM over the reply minus its outer padding. Per-chunk padding sits inside
                # this span and stays counted, because those pauses are part of the delivery.
                "rate": words / (spoken_s / 60) if spoken_s else 0.0,
                "seam": max(measured) if measured else float("nan"),
                "seams": len(measured),
                "seams_missing": len(seams) - len(measured),
                "chunks": len(audio_parts),
                "catchup": (synth_ms / 1000) / dur if dur else 0.0,
                "lead": lead if lead is not None else 0.0,
                "trail": trailing_silence(audio, rate),
                "headroom": float(np.abs(audio).max()) if len(audio) else 0.0,
                "audio_s": dur,
                "words": words,
            })

    if not rows:
        return {"preset": name, "error": "no audio produced"}

    def med(key):
        return statistics.median(r[key] for r in rows)

    seams = [r["seam"] for r in rows if r["seam"] == r["seam"]]
    missing = sum(r["seams_missing"] for r in rows)
    out = {
        "preset": name,
        "voice": voice,
        "speed": speed,
        "min_words": min_words,
        "debounce": debounce,
        "gap": gap,
        "n": len(rows),
        "rate": round(med("rate"), 1),
        # NaN propagates rather than defaulting to 0.0, so a preset that never produced a
        # measurable join reports as failed rather than as a flawless score.
        "seam": round(max(seams), 4) if seams else float("nan"),
        "seams_measured": len(seams),
        "seams_missing": missing,
        "chunks": round(med("chunks"), 1),
        "catchup": round(med("catchup"), 3),
        "lead_ms": round(med("lead"), 1),
        "trail": round(med("trail"), 3),
        "headroom": round(med("headroom"), 3),
        "audio_s": round(med("audio_s"), 2),
    }
    out["problems"] = check(out)
    return out


def check(row: dict) -> list[str]:
    problems = []
    low, high = TARGETS["rate"]
    if not low <= row["rate"] <= high:
        problems.append(f"rate {row['rate']} wpm outside {low}-{high}")
    if not row["seams_measured"]:
        problems.append("no intonation seam could be measured - the join is unverified")
    elif row["seam"] > TARGETS["seam"][1]:
        problems.append(f"intonation seam {row['seam']:.3f} > {TARGETS['seam'][1]}")
    if row["catchup"] > TARGETS["catchup"][1]:
        problems.append(
            f"synthesis {row['catchup']:.2f}x realtime - cannot keep ahead of the speaker"
        )
    if row["trail"] > TARGETS["trail"][1]:
        problems.append(f"trailing silence {row['trail']:.2f}s")
    if row["headroom"] > TARGETS["headroom"][1]:
        problems.append(f"peak {row['headroom']} pinned at full scale")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeat", type=int, default=1, help="passes over the corpus (median is reported)")
    ap.add_argument("--json", type=Path, help="write the full table here")
    ap.add_argument("--backend", default=config.TTS_BACKEND, help="tts backend: kokoro or piper")
    args = ap.parse_args()

    config.TTS_BACKEND = args.backend
    print(f"backend {config.TTS_BACKEND} | voice synthesised on CPU | "
          f"{len(CORPUS)} passages x {args.repeat}\n")
    print(f"{'preset':<22} {'wpm':>5} {'seam':>6} {'joins':>7} {'catchup':>8} "
          f"{'lead ms':>8} {'trail':>6} {'peak':>5}")
    print("-" * 84)

    rows = []
    for preset in PRESETS:
        row = measure(preset, args.repeat)
        rows.append(row)
        if "error" in row:
            print(f"{preset[0]:<22} {row['error']}")
            continue
        seam_txt = "  n/a" if row["seam"] != row["seam"] else f"{row['seam']:6.3f}"
        joins = f"{row['seams_measured']}/{row['seams_measured'] + row['seams_missing']}"
        print(f"{row['preset']:<22} {row['rate']:>5.0f} {seam_txt:>6} {joins:>7} "
              f"{row['catchup']:>8.2f} {row['lead_ms']:>8.0f} {row['trail']:>6.2f} "
              f"{row['headroom']:>5.2f}")
        for problem in row["problems"]:
            print(f"{'':<22}   ! {problem}")

    # A preset that errored has no "problems" list at all, so it must be excluded by name:
    # otherwise a preset that produced no audio counts as inside every target and the run
    # still exits 0 - a perfect score for a total failure.
    clean = [r for r in rows if not r.get("error") and not r.get("problems")]
    print("-" * 84)
    print(f"{len(clean)}/{len(rows)} presets inside every target")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
