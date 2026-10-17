"""Tunable settings. Every value can be overridden by an env var of the same name."""

import os

SAMPLE_RATE = 16000
CHUNK = 1600  # 100 ms per read

# Recording / endpointing
MAX_RECORD_SECONDS = 30
SILENCE_SECONDS_TO_STOP = 0.8
ONSET_SECONDS = 0.15
MIN_SPEECH_SECONDS = 0.4
NOISE_MARGIN = 2.2
MIN_THRESHOLD = 0.006

# Internet access for the model. Off by default; when on, the agent can search the web and
# read pages. It is the only thing in the agent that talks to the internet at runtime.
WEB_ENABLED = os.environ.get("WEB_ENABLED", "1") not in ("0", "false", "no", "")
WEB_TIMEOUT = float(os.environ.get("WEB_TIMEOUT", "10"))
WEB_MAX_CHARS = int(os.environ.get("WEB_MAX_CHARS", "4000"))
WEB_MAX_ROUNDS = int(os.environ.get("WEB_MAX_ROUNDS", "4"))

# Speech to text: "whisper" (local, free) or "google" (free, needs internet)
STT_BACKEND = os.environ.get("STT_BACKEND", "whisper")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base.en")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.environ.get("WHISPER_COMPUTE", "int8")
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "en")

# Brain: any Ollama-served model
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:3b-instruct")
LLM_KEEP_ALIVE = os.environ.get("LLM_KEEP_ALIVE", "30m")  # avoid reloading the model
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.6"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "160"))
LLM_NUM_CTX = int(os.environ.get("LLM_NUM_CTX", "2048"))

# Voice: "kokoro" (neural, best quality), "piper" (neural, lighter),
# "sapi" (Windows voices) or "silent"
TTS_BACKEND = os.environ.get("TTS_BACKEND", "kokoro")
KOKORO_VOICE = os.environ.get("KOKORO_VOICE", "af_bella")
KOKORO_MODEL = os.environ.get("KOKORO_MODEL", "kokoro-v1.0.onnx")
PIPER_VOICE = os.environ.get("PIPER_VOICE", "en_US-lessac-high")
NEURAL_LENGTH_SCALE = float(os.environ.get("NEURAL_LENGTH_SCALE", "1.05"))  # >1 = slower
NEURAL_NOISE_SCALE = float(os.environ.get("NEURAL_NOISE_SCALE", "0.6"))
NEURAL_NOISE_W_SCALE = float(os.environ.get("NEURAL_NOISE_W_SCALE", "0.75"))
NEURAL_GAP_SECONDS = float(os.environ.get("NEURAL_GAP_SECONDS", "0.05"))  # breath between chunks
NEURAL_MIN_CHUNK_WORDS = int(os.environ.get("NEURAL_MIN_CHUNK_WORDS", "4"))
NEURAL_FINAL_DEBOUNCE = float(os.environ.get("NEURAL_FINAL_DEBOUNCE", "0.28"))
TTS_RATE = int(os.environ.get("TTS_RATE", "0"))  # sapi only: -10 slow .. 10 fast
TTS_VOICE = os.environ.get("TTS_VOICE", "")  # sapi only: substring of the voice name

# Conversation
MAX_TOOL_ROUNDS = 4
HISTORY_TURNS = 6

# Measured, not guessed. qwen2.5:3b reliably stops calling tools when the system prompt
# grows past a few lines, even though the tool instruction is present and correct: with the
# previous, longer prompt, "what time is it" and "19.99 * 1.2 + 3" produced zero tool calls
# (0/3 each) while the same questions against this shorter prompt call them 3/3. Two specific
# clauses were each individually responsible for suppressing tool use - the original
# "say numbers the way a person would say them out loud rather than as digits" (which the
# model obeyed by verbalising digits itself instead of calling the tool) and a longer
# "never mention that you used a tool" instruction (which read as "do not use tools").
#
# Consequence: keep this prompt short, and keep every style requirement that measurably
# suppresses tool calling out of it. evals/golden.jsonl has tool-call cases so a regression
# here fails the suite instead of reaching your ears.
SYSTEM_PROMPT = (
    "You are a voice assistant. Everything you write is read aloud, so talk like a person: "
    "one to three short sentences, no markdown, lists, emoji, code or URLs.\n"
    "Use a tool first whenever the answer needs the current date or time, weather, prices, "
    "news, or arithmetic.\n"
    "Be warm and natural. Vary your openers. Never mention tools or that you are an AI.\n"
    "Answer only what was asked, then stop talking. Do not finish by offering more help, asking "
    "what else they need, or asking a question."
)