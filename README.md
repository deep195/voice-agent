# Local voice agent

A push-to-talk assistant that runs on your own machine. **Whisper** hears you, a local
**Ollama** model thinks, **Kokoro** speaks. No API keys, no account, no per-token bill.

```
ENTER  ->  mic  ->  faster-whisper  ->  ollama (+ tools)  ->  kokoro  ->  speakers
```

It is deliberately built out of parts that fit on a small laptop. The trade-offs that
implies — and the ones where it is still short — are measured in
[evals/BENCHMARKS.md](evals/BENCHMARKS.md) rather than asserted here.

---

## Quick start

**1. Ollama**, the only external piece:

```powershell
ollama serve
ollama pull qwen2.5:3b-instruct
```

**2. Python side** — creates `.venv` and installs everything:

```powershell
setup.bat
```

The first run also fetches the Whisper model (~150 MB) and Kokoro plus its voices (~340 MB).
Both cache under `~\.cache` and `~\.kokoro`, so later starts are instant.

**3. Talk:**

```powershell
run.bat
```

Press **ENTER** and speak. It stops recording when you go quiet, then answers out loud.
Press **ENTER** again while it talks to cut it off.

| key | action |
| --- | --- |
| `ENTER` | start talking |
| `ENTER` while it talks | interrupt |
| `q` + ENTER | quit |
| `r` + ENTER | clear conversation memory |
| `w clip.wav` + ENTER | record until you press ENTER, save a wav |

---

## The dashboard

```powershell
run-ui.bat
```

Opens <http://127.0.0.1:8765>: a live view of whether it is hearing you or talking, the
transcript and reply side by side, and **every voice parameter changeable while it is still
speaking** — voice, pacing, phrase length, pause before ending, breath between phrases, and
the model itself. Nothing restarts; the next sentence already uses the new value, and
*play sample* previews a voice without spending a turn.

Recording works either way, switchable in the panel:

- **this browser** — the microphone is captured in the page. Hold SPACE or the button.
- **the app** — Python records with the same endpointing as the terminal version, for when a
  browser blocks the mic. Click once, speak, and it stops on its own.

No web framework and no build step: the server is Python's own `http.server`.

> **Keep it on `127.0.0.1`.** There is no login and no rate limiting. Anything on your
> network could open your microphone and read your files.

---

## How it sounds less like a machine

Chunked speech sounds robotic when every chunk is synthesised as a finished sentence:
intonation restarts and the pauses land like seams. Three things address that, all in
`agent/tts.py`:

- **Phrases, not sentences.** Text is cut at clause punctuation after a few words, so the
  voice receives whole phrases (`stream_chunks`).
- **Continuous prosody.** A chunk that is not the reply's end loses its trailing full stop,
  so the voice keeps going instead of dropping at every boundary. The final chunk waits
  ~0.3 s to find out whether more is coming, and gets a proper falling ending if not.
- **Speakable text.** `normalize()` turns `50°F`, `30%`, `Dr.`, `etc.` into words, and
  `strip_signoffs()` removes "Let me know if you need anything else" and its relatives, so
  the voice never trips over a symbol or signs off every sentence.

Synthesis runs on its own thread, ahead of playback, so the agent begins speaking on the
first finished chunk instead of pausing to generate.

**One caveat, measured.** On a small CPU, phrase length decides whether the voice keeps up
with its own playback, because each synthesis call costs about 598 ms before a single sample
is produced. Four-word chunks spend most of their time on that overhead and the voice runs
slower than real time; from about twelve words up it stays ahead. The cost is paid at the
front — roughly 2.9 s to first audio at four words versus 5.9 s at twelve. There is no
setting that both starts fast and stays ahead on four cores. See
[the benchmark](evals/BENCHMARKS.md#chunk-size-is-the-variable-that-decides-whether-the-voice-keeps-up).

---

## Internet access

A 3B model has a training cutoff and no idea what happened this morning, so current questions
go to DuckDuckGo and the agent reads a page before answering. No key, no account, no cost —
the results page is parsed directly in `agent/tools.py`, which is why it can occasionally be
rate-limited or change shape.

```
WEB_ENABLED=1        set 0 and it never touches the network at all
WEB_MAX_CHARS=4000   text kept from a page
WEB_TIMEOUT=10       seconds to wait
WEB_MAX_ROUNDS=4     searches per turn
```

Two things it will not do: execute code from a fetched page, or send anything about you. A
fetch is read-only, and the only thing ever sent is the query the model itself wrote.

The trade-off is plain: **queries reach DuckDuckGo exactly as you said them.** With
`WEB_ENABLED=0` the agent is fully offline and cannot answer anything current.

---

## Checking it still works

Two suites, because "did it say the right thing" and "did it sound right" are different
questions.

### Behaviour

27 turns covering tools, context, identity, voice hygiene, web answers and awkward input.

```powershell
.venv\Scripts\python.exe evals\run_eval.py --no-speak
```

```powershell
--id time_now math_add      run specific cases
--tags tool:web             run a group
--listen                    transcribe the replies too (slow; the whole loop)
--repeat 3                  catch flaky behaviour
--baseline old.jsonl        compare against an earlier run
```

Runs are greedy by default (temperature 0) so two runs of the same code agree and a baseline
means something; the app itself runs at 0.6 for variety. Every run appends to
`evals/results.jsonl`.

Across three consecutive runs on the laptop below: **25–27 of 27**, median turn 6.0–7.7 s.
The variance is machine load, not flakiness in the assertions. Two cases are genuinely
marginal:

- `no_urls` fails about half the time at 95 spoken words against a limit of 80. The model
  over-explains where the test wants brevity. Fixing it means making a 3B model terser, not
  loosening the check.
- `long_question` trips its 40 s latency ceiling when the machine is busy. A ~40 s turn is a
  real experience on a CPU this size, so it is reported rather than timed around.

Add a case as one JSON object per line in `evals/golden.jsonl`:

```json
{"id": "my_case", "tags": ["knowledge"], "utterance": "how many continents are there",
 "expect": {"contains_any": ["7", "seven"]}, "max_reply_words": 25}
```

### Voice

`evals/voice_eval.py` measures the audio itself, because a transcript hides exactly the
failures that matter: a seam where two chunks join, synthesis that cannot stay ahead of the
speaker, trailing dead air, a waveform pinned at full scale.

```powershell
.venv\Scripts\python.exe evals\voice_eval.py
```

| metric | measures | target |
| --- | --- | --- |
| `rate` | words per minute of speech | 150–175 |
| `seam` | pitch discontinuity at a chunk join | < 0.12 |
| `catchup` | synthesis time ÷ audio duration | ≤ 1.0 |
| `lead` | time to first playable chunk | — |
| `trail` | trailing silence | < 0.12 s |
| `headroom` | peak level | < 0.98 |

It exits non-zero unless some preset clears every target, so it works as a check and not
just a report. On that laptop **no preset currently passes every target**, which is the
finding rather than a broken harness.

---

## Performance, honestly

Full numbers, sources and method: **[evals/BENCHMARKS.md](evals/BENCHMARKS.md)**.

Measured on an i7-8565U / 8 GB laptop with no GPU in the loop:

| | |
| --- | --- |
| Time to first spoken token | **4.1 s** median |
| Whole turn finished | **6.0–7.7 s** median |
| Turn p90 / worst | 19.7–26.2 s / 34.3–40.1 s |
| Kokoro synthesis | 1.12x faster than real time, marginal cost |

For context, Amazon's Nova Sonic technical report (attributing the data to Artificial
Analysis) puts **time to first audio** for hosted models at a median of 1.05 s for Nova
Sonic, 1.11 s for GPT-4o Realtime and 1.36 s for GPT-4o-mini Realtime, and Amazon's later
Nova 2 report puts Gemini 2.5 Flash Live at 0.63 s. Those are datacenter GPUs running
different models, so this is not a like-for-like
comparison — but the honest summary is that pressing space and waiting ~5–6 seconds for the
first sound is not the experience those figures describe, and the gap is **cores**, not
architecture. Nothing in this stack uses the GPU.

Two notes on numbers people repeat:

- Kokoro is often described as "1.3x faster than real time". That is its *marginal* rate and
  it is accurate — but it says nothing about short chunks, where a fixed ~598 ms per call
  dominates and the voice drops behind playback.
- Benchmark figures for Piper are easy to overstate in either direction by forgetting to warm
  the session first. Medium voices measured ~10x real time here; `lessac-high` ~1.5–1.9x.

---

## How it is put together

| file | role |
| --- | --- |
| `main.py` | the loop: record → transcribe → reply → speak |
| `agent/audio.py` | mic capture, adaptive noise floor, silence endpointing |
| `agent/stt.py` | faster-whisper locally, or Google's free endpoint |
| `agent/llm.py` | streaming chat client for Ollama (stdlib only) |
| `agent/tts.py` | chunking, normalisation, Kokoro/Piper/SAPI, playback |
| `agent/tools.py` | `current_time`, a safe `calculate`, `web_search`, `web_fetch` |
| `agent/server.py` | dashboard backend — stdlib HTTP and SSE |
| `web/` | the dashboard: `index.html`, `app.js`, `style.css` |
| `agent/config.py` | every setting |
| `evals/` | `golden.jsonl`, `run_eval.py`, `voice_eval.py`, `BENCHMARKS.md` |

### Options

```powershell
run.bat --voice af_heart                    # another of the 54 Kokoro voices
run.bat --tts piper --voice en_US-amy-medium    # ~10x faster, lower quality
run.bat --speed 1.1                         # slower, more deliberate
run.bat --stt google                        # free Google STT, no download
run.bat --whisper-model small.en            # better STT, slower
run.bat --model qwen2.5:7b-instruct         # smarter, far slower on a laptop
run.bat --no-memory                         # forget context each turn
run.bat --tts sapi                          # built-in Windows voice
run.bat --list-devices
```

### Voices

**Kokoro** (default) — 82 M parameters, CPU-only, the most polished English voice available
at this size. `af_bella` is the default; `af_heart` is the warmest. Also `af_alloy`,
`af_aoede`, `af_jessica`, `af_kore`, `af_nicole`, `af_nova`, `af_river`, `af_sarah`, `af_sky`,
`am_adam`, `am_echo`, `am_eric`, `am_fenrir`, `am_liam`, `am_michael`, `am_onyx`, `am_puck`,
`am_santa`, plus British (`bf_*`, `bm_*`), Spanish (`ef_*`, `em_*`) and French (`ff_siwis`).
The int8 build is *slower* on CPU, so this downloads float32 (~310 MB).

**Piper** — lighter and much faster. `en_US-lessac-high` (default), `en_US-amy-medium`,
`en_US-ryan-medium`, `en_US-joe-medium`, `en_US-kristin-medium`, `en_US-danny-low`,
`en_US-kathleen-low`, `en_GB-alan-medium`, `en_GB-jenny_dioco-medium`,
`hi_IN-pratham-medium`. Any `rhasspy/piper-voices` model works.

### Tuning

`agent/config.py` holds everything; most of the names below are env vars too, the rest are
edited in the file.

- Voice: `KOKORO_VOICE`, `KOKORO_MODEL`, `PIPER_VOICE`, `NEURAL_LENGTH_SCALE`
- Delivery: `NEURAL_MIN_CHUNK_WORDS`, `NEURAL_FINAL_DEBOUNCE`, `NEURAL_GAP_SECONDS`,
  `NEURAL_NOISE_SCALE`, `NEURAL_NOISE_W_SCALE` (Piper only)
- Listening: `SILENCE_SECONDS_TO_STOP` (0.8), `ONSET_SECONDS`, `NOISE_MARGIN`,
  `MAX_RECORD_SECONDS` (30)
- Model: `LLM_MODEL`, `LLM_MAX_TOKENS`, `HISTORY_TURNS`, `MAX_TOOL_ROUNDS` (4)
- Web: `WEB_ENABLED` and the `WEB_*` knobs above

One knob is deliberately absent. There is no prompt setting for reply length or for
suppressing the "how can I assist you today?" habit — both are handled in code
(`agent/tts.py`: `NEURAL_LENGTH_SCALE` and `strip_signoffs`), because every prompt edit strong
enough to fix them also stopped the 3B model calling its tools, which is a much worse failure.
The system prompt stays short on purpose.

---

## Notes and limits

- **Licensing.** `qwen2.5:3b-instruct` ships under the **Qwen Research** licence, not
  Apache-2.0 — the 0.5B, 1.5B and 7B sizes are Apache-2.0. Check the terms before anything
  commercial. Whisper, Kokoro and Piper carry their own licences.
- **Tool calling is the weak spot.** Qwen2.5-3B's multi-turn function-calling scores are poor
  next to the 7B. Single tools such as `calculate` and `current_time` work well; chained
  requests are where it degrades.
- Windows is the tested platform. `--tts sapi` uses Windows voices; on Linux and macOS the
  fallback goes through `pyttsx3`, which usually needs `espeak`.
- STT accuracy: `--whisper-model small.en` hears you better and transcribes slower.
- A web search that has to wait on a page can take 25–45 s. If the *first* reply after
  starting the app is slow, that is the model being loaded into memory, not a bug.
