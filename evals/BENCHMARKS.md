# Benchmarks

Everything under "Measured here" was produced on this repository's own machine by
`evals/voice_eval.py` and `evals/run_eval.py`. Everything under "Published figures" is
someone else's measurement, reproduced here with its source and its caveats. The two are
kept apart on purpose: the published numbers come from datacenter GPUs running vendor
models, and presenting them next to laptop-CPU numbers without that distinction would be
disguising a hardware gap as a quality gap.

## The machine these numbers came from

| | |
|---|---|
| CPU | Intel Core i7-8565U @ 1.80 GHz, 4 cores / 8 threads |
| RAM | 7.9 GB |
| GPU | NVIDIA GeForce MX150, integrated Intel UHD 620 (nothing here is used) |
| OS | Windows 11 |
| Runtime | Python 3.14.5, all inference on CPU |

An 8 GB laptop from 2018, and that constraint explains most of what follows.

## Measured here

### Where the time goes in a turn

Median of a six-turn probe, following tool calls through to spoken text:

| stage | median | note |
|---|---|---|
| Time to first spoken token | 4.1 s | a 3B model on CPU; this is the floor on responsiveness |
| Whole turn finished | 7.0 s | includes running tools |
| Generation rate | 12 chars/s | |

### Voice synthesis

Kokoro on CPU, splitting synthesis cost into its fixed and marginal parts gives the whole
story:

| | |
|---|---|
| Fixed cost per synthesis call | ~598 ms |
| Marginal cost | 0.756 s per second of audio (1.32x faster than real time) |

Both rows come from the same least-squares fit of synthesis time against audio length, so
they are comparable to each other. The thread experiment below is measured differently —
whole-passage realtime factor — and putting the two side by side would make a tuning change
look like a regression when it is not.

Turning ONNX threads up from its default is worth about 3% on this chip (median realtime
factor 0.84 at the default, 0.82 at eight threads), and pinning it to a single thread is
clearly worse at 1.31. That is a rounding error next to the chunk-size effect below, and the
real multiple on this machine is not threads.

That 1.32x figure is the honest version of "1.3x faster than realtime on CPU". It is real,
but it is the *marginal* rate, and it is the rate at which audio lengthens as you speak
more. It says nothing about short chunks, because a synthesis call has to happen whether or
not it produces much audio.

### Backend comparison

Median of five timed calls over the same passage, after two warmups each. The warmup matters
more than it looks: a session that is only warmed once reports roughly twice the real cost,
which is how an earlier run of this table came to say Piper was slower than it is.

| engine | rate | realtime factor |
|---|---|---|
| Piper `en_US-ryan-medium` | 22.05 kHz | 9.5x faster than real time |
| Piper `en_US-amy-medium` | 22.05 kHz | 9.7x faster than real time |
| Piper `en_US-lessac-high` | 22.05 kHz | 1.5–1.9x faster than real time |
| Kokoro `af_bella` | 24 kHz | 1.12x faster than real time |

Kokoro costs roughly an order of magnitude more CPU than a medium Piper voice. That is the
price of the quality it is here for, and on a CPU-bound laptop it is worth knowing when the
voice starts falling behind.

### Chunk size is the variable that decides whether the voice keeps up

Every other setting held still, `af_bella` at speed 1.20:

| words per chunk | synthesis vs real time | time to first audio | intonation seam | wpm |
|---|---|---|---|---|
| 4 (default) | 1.25x — **slower than real time** | 2.9 s | 1.85 | 210 |
| 5 | 1.12x | 2.9 s | 0.23 | 182 |
| 8 | 1.03x | 4.3 s | 0.16 | 196 |
| 12 | **0.94x** | 5.9 s | 0.13 | 186 |
| 18 | 0.89x | 6.1 s | 0.13 | 186 |
| 24 | 0.86x | 6.3 s | unmeasurable | 172 |

Under about eight words per chunk the fixed 598 ms dominates, synthesis runs slower than the
speaker, and the voice dries up mid-reply. From twelve words up it keeps ahead. The price is
paid at the front: waiting for a longer phrase before speaking roughly doubles the delay
before the user hears anything, from 2.9 s to 5.9 s.

This is the central trade-off on this hardware, and it is a genuine dilemma rather than a
setting to tune away. There is no chunk size that both starts fast and stays ahead of
playback on four cores.

The intonation seam is the pitch discontinuity where two independently synthesised chunks
join, as a fraction of pitch, measured by autocorrelation on the voiced audio either side of
each join. Smaller chunks make it worse, not better, because each chunk restarts the
prosody: at four words the worst join jumps 185% of pitch, at eighteen it is 13%.

Unmeasurable joins are reported rather than scored. A chunk that ends on a voiceless
consonant genuinely has no pitch to measure, and quietly treating that as a pass is how a
broken metric reports a perfect score.

### Behavioural suite

27 cases, `--no-speak`, deterministic at temperature 0, three consecutive runs on this
machine:

| run | passed | median turn | p90 | slowest |
|---|---|---|---|---|
| 1 | 26/27 | 6.7 s | 23.4 s | 40.1 s |
| 2 | 25/27 | 7.7 s | 26.2 s | 40.0 s |
| 3 | 27/27 | 6.0 s | 19.7 s | 34.3 s |

The spread is machine load, not unstable assertions. Two cases sit close to their limits and
are worth naming rather than hiding:

- `no_urls` fails in roughly half of runs, at 95 spoken words against a limit of 80. The model
  over-explains where the test wants brevity. It is a real weakness of a 3B model.
- `long_question` hits its 40 s ceiling when the laptop is busy. A 40 s turn is an accurate
  description of this hardware, so it is left in the suite as a standing reminder.

## Published figures

Reproduced with sources. Not comparable to the local numbers above without saying so: these
are vendor-hosted models on datacenter hardware, and the sources are the vendors' own
technical reports.

### Time to first audio

From query completion to first response audio, median, as published by Amazon in its Nova
Sonic and Nova 2 technical reports (both attribute the data to Artificial Analysis, a third
party — so it is independent measurement, but it is still not independently reproducible):

| model | TTFA median | TTFA mean | TTFA p90 |
|---|---|---|---|
| Gemini 2.5 Flash Live | 0.63 s | — | — |
| Nova Sonic | 1.05 s | 1.09 s | 1.35 s |
| GPT-4o Realtime | 1.11 s | 1.18 s | 1.68 s |
| GPT-4o-mini Realtime | 1.36 s | 1.29 s | 1.90 s |
| Gemini 2.0 Flash (experimental) | 1.24 s | 1.41 s | 2.24 s |

The Gemini 2.5 Flash Live figure is from Amazon's later Nova 2 report, alongside GPT-Realtime
at 0.98 s and Nova 2 Sonic at 1.39 s.

- <https://cdn.amazon.science/86/bb/4316d28940bd9a719abb28f45aaf/amazon-nova-sonic-technical-report-and-model-card-6-12.pdf>
- <https://cdn.amazon.science/c5/3d/84514a224666b5be6de4b43ef4aa/nova-2-0-technical-report2.pdf>

### The gap, stated carefully

Local time to first audio is **2.9 s** with default chunking, and the model's own time to
first token before that is a median **4.1 s**.

A user pressing space and hearing nothing for several seconds is not the experience those
cloud figures describe. The honest comparison is:

| | this repo, laptop CPU | market leaders, datacenter |
|---|---|---|
| Model starts talking | 4.1 s median | folded into the 0.6–1.4 s TTFA |
| First audio | 2.9 s after that (4-word chunks) | 0.63–1.36 s from the end of the question |
| Speech keeps up with playback | only at 12+ words per chunk | yes, throughout |

So the gap to Gemini 2.5 Flash Live is roughly **5–6 seconds of silence before the first
sound**, and it is a hardware gap rather than a modelling one. The same `af_bella` voice on
the same code path runs on a recent desktop CPU at a fraction of these times; what is missing
here is cores, and a GPU that the stack does not use.

Two things would close most of it, and neither is a rewrite: give ONNX the machine's
remaining threads (measured above, worth about 3%), and move the model to hardware with more
cores, which is where the real multiple lives.

## Reproducing

```powershell
.venv\Scripts\python.exe evals\voice_eval.py                     # voice: rate, seams, throughput
.venv\Scripts\python.exe evals\voice_eval.py --repeat 3 --json out.json
.venv\Scripts\python.exe evals\run_eval.py --no-speak            # behaviour, no audio
.venv\Scripts\python.exe evals\run_eval.py --baseline previous.jsonl
```

`voice_eval.py` exits non-zero unless some preset is inside every target, so it works as a
check and not only as a report. Targets are stated in `TARGETS`: 150–175 wpm, seam under
0.12, synthesis at or below real time, under 0.12 s of trailing silence, peak below 0.98.

On this machine no preset currently satisfies all of them, which is the finding, not a
misconfiguration of the harness.
