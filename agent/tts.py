"""Text to speech.

Kokoro (neural, offline) is the default, Piper is the lighter neural option: a synthesis
thread runs ahead of a playback thread, so speech starts on the first finished sentence and
never waits on synthesis. SAPI (--tts sapi) is the built-in fallback, silent is for tests.
"""

import queue
import re
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

from . import config

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
CLAUSE_BREAK = re.compile(r"[.!?,;:]\s+")
TERMINAL = re.compile(r"[.!?]+\s*$")
REPLACEMENTS = [
    ("°F", " degrees fahrenheit"),
    ("°C", " degrees celsius"),
    ("°f", " degrees fahrenheit"),
    ("°c", " degrees celsius"),
    ("°", " degrees "),
    ("%", " percent"),
    ("&", " and "),
    ("@", " at "),
    ("+", " plus "),
    ("=", " equals "),
    ("$", " dollars "),
    ("~", " about "),
    ("etc.", "et cetera"),
    ("e.g.", "for example"),
    ("i.e.", "that is"),
    ("vs.", "versus"),
    ("Mr.", "Mister"),
    ("Mrs.", "Missus"),
    ("Dr.", "Doctor"),
    ("St.", "Saint"),
]
SPV_PURGE = 3  # SAPI flag: stop the current utterance immediately
PLAY_BLOCK = 2048  # samples per write, so skipping is near instant

PRONOUNCED = {
    "\u2014": ", ",  # em dash -> a pause
    "\u2013": " - ",  # en dash
    "\u2026": "...",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u00a3": " pounds ",
    "\u20ac": " euros ",
    "\u00a5": " yen ",
    "\u00b0": " degrees ",
}


def normalize(text: str) -> str:
    """Make text sayable: symbols become words, odd characters disappear."""
    for pattern, replacement in REPLACEMENTS:
        text = text.replace(pattern, replacement)
    # Dashes and smart punctuation are stripped as non-ascii later, which silently welded
    # words together ("place—the" -> "placethe"). Give them a spoken form first.
    for pattern, replacement in PRONOUNCED.items():
        text = text.replace(pattern, replacement)
    # Markdown link syntax becomes just the label, then the bare URL is dropped: turning it
    # into " link " made the assistant read out search result lists, which is the one thing
    # a voice answer should never do. Sources stay visible in the UI transcript.
    # One level of nesting: a Wikipedia URL like .../Gulf_(disambiguation) contains its own
    # parentheses, and a flat [^)]* leaves a stray ")" that the voice then reads out.
    text = re.sub(r"\[([^\]]*)\]\(\s*(?:[^()]|\([^()]*\))*\s*\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"```.*?```", " code block omitted ", text, flags=re.S)
    # Numbered lists need two different treatments depending on what they are. Must run
    # before the whitespace collapse below, while the newlines still exist.
    # A list introduced as sources is reference material: drop it.
    LIST = r"\n\s*\d+\s*[.):-][^\n]*(?:\s*\n+\s*\d+\s*[.):-][^\n]*)+"
    match = re.search(LIST, text)
    lead_in = text[: match.start()] if match else ""
    if re.search(r"(?:sources?|references?|links?|headlines|articles?)[^.!?:]*:\s*\Z", lead_in, re.I):
        # Reference material, introduced as sources: the list is dropped, because reading
        # out citation labels and addresses is the classic chatbot tell.
        text = re.sub(LIST, " ", text)
        without = re.sub(
            r"[^.!?]*\b(?:sources?|references?|links?|headlines|results?|articles?)\b[^.!?]*:\s*$",
            "",
            text,
            flags=re.I,
        )
        if len(without.split()) > 3:
            text = without
        else:
            # There was no prose, only a link dump - which is what happens when the model
            # answers "give me the latest news" by listing headlines. Dropping the list then
            # leaves "Here are the latest articles:" and nothing else, so the headlines are
            # the answer: keep them, minus the URLs and the lead-in.
            text = lead_in + match.group(0)
            text = re.sub(
                r"[^.!?]*\b(?:sources?|references?|links?|headlines|results?|articles?)\b[^.!?]*:\s*",
                "",
                text,
                flags=re.I,
            )
            # Split on the newlines, not on ". " - the numbering contains periods too.
            # Three items is about as much of a list as anyone wants to hear.
            items = [
                re.sub(r"^\s*\d+\s*[.):-]\s*", "", line).strip()
                for line in text.splitlines()
                if line.strip()
            ]
            # An item that already ends in a full stop must not gain a second one.
            text = ". ".join(item if item.endswith(".") else item + "." for item in items[:3])
    else:
        # Otherwise the numbering is decoration and the items are the answer. Reading "one
        # point two, two point three" is the classic way a list betrays it came from a
        # chatbot, so the markers go and the items become one spoken enumeration.
        text = re.sub(r"\n\s*\d+\s*[.):-]\s*", ", ", text)
        # The lead-in kept its colon and now runs straight into the list: "are:, 2, 3, 5".
        text = re.sub(r"\s*:\s*,\s*", ": ", text)
        # An item that already ended a sentence must not be followed by a comma.
        text = re.sub(r"([.!?])\s*,\s*", r"\1 ", text)
    text = re.sub(r"(?<=\w)_(?=\w)", " ", text)  # af_bella, my_script -> two words
    text = re.sub(r"[*_`#>]", "", text)
    text = "".join(c for c in text if c.isprintable() or c in " \n\t")
    text = text.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", text).strip()


SIGNOFF_PATTERNS = [
    # Must be able to sit on its own: the leading "if you need any help," is only a
    # sign-off tail when something follows it.
    r"(?:if|should)\s+you\s+(?:have|need|want)\s+"
    r"(?:any\s+|more\s+|any\s+more\s+|any\s+other\s+|another\s+|further\s+|some\s+)*"
    r"(?:help|question|assistance|information|detail|idea|thought|need)[^.!?]*",
    # "If you have another question..." and "and I'll be happy to help" style tails.
    r"(?:and\s+i'?ll\s+be\s+happy\s+to\s+help|i'?ll\s+be\s+happy\s+to\s+help|i'?m\s+happy\s+to\s+help)[^.!?]*",
    r"(?:feel\s+free\s+to|don'?t\s+hesitate\s+to)\s+(?:ask|reach\s+out|contact|let\s+me\s+know)[^.!?]*",
    # These can legitimately be the whole reply, but strip_signoffs refuses to shrink a
    # reply below three words, so they only ever come off a longer sentence.
    r"(?:how|what)\s+(?:can|may|could)\s+i\s+(?:assist|help|support)\s+you[^.!?]*",
    r"is\s+there\s+(?:anything\s+(?:else|more)|more)\b[^.!?]*",
    r"(?:do|would)\s+you\s+(?:like|want)\s+(?:me\s+to\s+)?[^.!?]*",
    r"let\s+me\s+know\s+(?:if|whether)\s+you[^.!?]*",
    r"i'?m\s+here\s+(?:if|for)\s+you[^.!?]*",
]
# A sign-off clause is removed wherever it sits in the reply, except at position 0: a reply
# that is *only* "Would you like me to check?" is a legitimate answer and must survive.
# Not anchored to the end because models like to sandwich one - "Goodbye! Don't hesitate
# to ask if you need help in the future. Have a great day!"
SIGNOFFS = re.compile(
    r"(?<=[.!?])\s+(?:" + "|".join(SIGNOFF_PATTERNS) + r")\s*[.!?]?",
    re.IGNORECASE,
)


def strip_signoffs(text: str) -> str:
    """Drop the canned "how can I assist you today?" tail.

    Small models append one to almost every reply, and hearing it every turn is the most
    obvious tell that you are talking to a machine. Done here rather than in the system
    prompt because every prompt edit strong enough to suppress it was also strong enough to
    stop the model calling its tools (see agent/config.py and evals/golden.jsonl).

    Refuses to remove anything if what is left is not a substantial answer, so a reply that
    is only a sign-off does not collapse to "?" or "Sure..".
    """
    text = text.strip()
    for _ in range(4):  # replies sometimes stack two ("... How can I help? Let me know.")
        stripped = re.sub(r"\s+([.!?])", r"\1", SIGNOFFS.sub("", text)).strip()
        # Nothing audible left (a lone "?" or "Sure.") means the reply was only a
        # sign-off, so keep it rather than answer with silence.
        if not re.search(r"[a-z0-9]", stripped, re.IGNORECASE) or stripped == text:
            break
        text = stripped
    return re.sub(r"[\s,;]+([.!?])", r"\1", text).strip()


def for_speech(text: str, final: bool = True) -> str:
    """Clean up text for a voice, and set the prosody that ends the utterance.

    Non-final chunks lose their trailing full stop so the voice carries on instead of
    restarting intonation on every chunk - that is what makes chunked speech sound robotic.
    """
    text = strip_signoffs(normalize(text))
    if not text:
        return ""
    if final:
        if text[-1] not in ".!?,;:":
            text += "."
        return text
    text = TERMINAL.sub("", text).strip()
    return text if text else ""


def _find_cut(text: str, min_words: int) -> int | None:
    for match in CLAUSE_BREAK.finditer(text):
        if len(text[: match.end()].split()) >= min_words:
            return match.end()
    return None


def stream_chunks(chunks, min_words: int | None = None):
    """Group streamed text into speakable chunks, cutting at clause boundaries.

    Cuts only after a few words so the voice gets phrases, not stuttering fragments.
    The phrase length is re-read for every chunk so a dashboard slider change lands on the
    next chunk instead of waiting for a whole new reply.
    """
    buffer = ""
    for piece in chunks:
        buffer += piece
        while True:
            target = config.NEURAL_MIN_CHUNK_WORDS if min_words is None else min_words
            cut = _find_cut(buffer, target)
            if cut is None:
                break
            chunk = buffer[:cut].strip()
            buffer = buffer[cut:].lstrip()
            if chunk:
                yield chunk
    if buffer.strip():
        yield buffer.strip()


def stream_sentences(chunks):
    """Group streamed text chunks into sentences."""
    buffer = ""
    for chunk in chunks:
        buffer += chunk
        parts = SENTENCE_END.split(buffer)
        if len(parts) > 1:
            buffer = parts[-1]
            for sentence in parts[:-1]:
                sentence = sentence.strip()
                if sentence:
                    yield sentence
    if buffer.strip():
        yield buffer.strip()


def piper_model_path(voice: str) -> str:
    """Locate a Piper voice model, downloading it once from Hugging Face."""
    cache = Path.home() / ".piper"
    if (cache / f"{voice}.onnx").exists() and (cache / f"{voice}.onnx.json").exists():
        return str(cache / f"{voice}.onnx")

    language, name, quality = voice.split("-")
    filename = f"{language[:2]}/{language}/{name}/{quality}/{voice}.onnx"
    from huggingface_hub import hf_hub_download

    print(f"[tts] downloading piper voice '{voice}' (one time, ~60 MB)...")
    cache.mkdir(parents=True, exist_ok=True)
    for suffix in ("", ".json"):
        source = Path(hf_hub_download("rhasspy/piper-voices", filename + suffix))
        target = cache / f"{voice}.onnx{suffix}"
        if not target.exists():
            target.write_bytes(source.read_bytes())
    return str(cache / f"{voice}.onnx")


KOKORO_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
KOKORO_VOICES_FILE = "voices-v1.0.bin"


def kokoro_paths(model: str = config.KOKORO_MODEL) -> tuple[str, str]:
    """Locate the Kokoro model and voice pack, downloading them once.

    The model file has to come from the kokoro-onnx release: it feeds speed in as an int,
    which the community Hugging Face exports do not accept.
    """
    cache = Path.home() / ".kokoro"
    model_path = cache / model
    voices_path = cache / KOKORO_VOICES_FILE
    if model_path.exists() and voices_path.exists():
        return str(model_path), str(voices_path)

    import urllib.request

    cache.mkdir(parents=True, exist_ok=True)
    for name in (model, KOKORO_VOICES_FILE):
        target = cache / name
        if target.exists():
            continue
        print(f"[tts] downloading kokoro '{name}' (one time, ~{target.stat().st_size if target.exists() else 0})...")
        urllib.request.urlretrieve(f"{KOKORO_RELEASE}/{name}", target)
    return str(model_path), str(voices_path)


class _Engine:
    """A neural backend: text in, float32 samples out at self.rate."""

    rate = 24000

    def synth(self, text: str) -> np.ndarray:
        raise NotImplementedError


class _KokoroEngine(_Engine):
    def __init__(self, voice: str, speed: float):
        from kokoro_onnx import Kokoro

        self._kokoro = Kokoro(*kokoro_paths())
        self.voice = voice
        self.speed = max(0.5, min(2.0, 1.0 / speed))  # kokoro speed is inverted
        self._kokoro.create("ready", voice=self.voice, speed=self.speed, lang="en-us")

    def synth(self, text: str) -> np.ndarray:
        samples, self.rate = self._kokoro.create(
            text, voice=self.voice, speed=self.speed, lang="en-us"
        )
        return samples.astype(np.float32)


class _PiperEngine(_Engine):
    def __init__(self, voice: str, speed: float):
        from piper import PiperVoice, SynthesisConfig

        piper = PiperVoice.load(piper_model_path(voice))
        self.rate = piper.config.sample_rate
        self._piper = piper
        self._settings = SynthesisConfig(
            length_scale=speed,
            noise_scale=config.NEURAL_NOISE_SCALE,
            noise_w_scale=config.NEURAL_NOISE_W_SCALE,
        )
        piper.synthesize("ready", self._settings)  # warm the onnx session

    def synth(self, text: str) -> np.ndarray:
        clips = [c.audio_int16_array for c in self._piper.synthesize(text, self._settings)]
        if not clips:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(clips)
        return audio.astype(np.float32) / 32768.0


class Speaker:
    def __init__(self, backend: str = config.TTS_BACKEND, rate: int = config.TTS_RATE,
                 voice: str = config.TTS_VOICE, speed: float = config.NEURAL_LENGTH_SCALE,
                 sink=None):
        self.backend = backend
        self.rate = rate
        self.voice = voice or self._default_voice(backend)
        self.speed = speed
        self.sink = sink
        self.error: str | None = None
        self._failed = False
        self._seq = 0
        self._skip_seq = 0
        self._lock = threading.Lock()
        self._engine_lock = threading.Lock()
        self._synth_lock = threading.Lock()  # one engine.synth() at a time
        self._threads: list[threading.Thread] = []
        self._sentences: "queue.Queue" = queue.Queue(maxsize=64)
        self._clips: "queue.Queue" = queue.Queue(maxsize=64)
        self._synth_busy = threading.Event()
        self._speaking = threading.Event()
        self._ready = threading.Event()
        self._speak_async = None
        self._engine = None
        self._engine_key: tuple | None = None
        self._rate = config.SAMPLE_RATE
        # Chunks handed to say() that the synth thread has not finished yet. The queue
        # itself is empty while the synth thread *holds* a chunk waiting out the debounce,
        # so queue emptiness alone would let a reply be declared finished one chunk early
        # and that orphan would be spoken at the start of the next turn.
        self._outstanding = 0
        self._dropped = 0

    @staticmethod
    def _default_voice(backend: str) -> str:
        if backend == "kokoro":
            return config.KOKORO_VOICE
        if backend == "piper":
            return config.PIPER_VOICE
        return ""

    @property
    def speaking(self) -> bool:
        return self._speaking.is_set()

    @property
    def pending(self) -> bool:
        if self._failed:
            return False
        with self._lock:
            outstanding = self._outstanding
        return bool(
            outstanding
            or not self._sentences.empty()
            or not self._clips.empty()
            or self._synth_busy.is_set()
            or self._speaking.is_set()
        )

    @property
    def seq(self) -> int:
        """The newest chunk number handed out; a turn starts above this."""
        with self._lock:
            return self._seq

    def start(self) -> None:
        if self.backend in ("kokoro", "piper"):
            self._spawn(self._neural_synth)
            if self.sink is None:
                self._spawn(self._neural_play)
            self._ready.wait(timeout=120)
        elif self.backend == "sapi":
            self._spawn(self._sapi_loop)
            self._ready.wait(timeout=10)

    def _spawn(self, target) -> None:
        thread = threading.Thread(target=target, daemon=True, name=target.__name__)
        thread.start()
        self._threads.append(thread)

    def _put(self, q: "queue.Queue", item) -> None:
        """Never block the reply stream: drop the oldest item if we are behind."""
        while True:
            try:
                q.put_nowait(item)
                return
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass

    def _put_sentence(self, item) -> None:
        """Queue a chunk and count it as outstanding work until _emit finishes it."""
        with self._lock:
            self._outstanding += 1
        while True:
            try:
                self._sentences.put_nowait(item)
                return
            except queue.Full:
                try:
                    self._sentences.get_nowait()
                except queue.Empty:
                    continue
                # The evicted chunk will never be synthesized, so stop counting it.
                with self._lock:
                    self._outstanding -= 1
                    self._dropped += 1

    def _next_sentence(self):
        try:
            return self._sentences.get(timeout=0.2)
        except queue.Empty:
            return "empty"

    # ---------- neural (kokoro, piper) ----------

    def reconfigure(self, voice: str | None = None, speed: float | None = None) -> None:
        """Change voice or pacing while running. The new engine loads on the next chunk."""
        with self._lock:
            if voice:
                self.voice = voice
            if speed is not None:
                # Clamped: the phonemizer saturates, so a wild value would be silently
                # ignored while the dashboard still displayed it.
                self.speed = min(2.0, max(0.5, float(speed)))

    def preview(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize a one-off sample with the live settings, without playing it."""
        with self._synth_lock:
            engine = self._current_engine()
            audio = engine.synth(text)
            return audio, engine.rate

    def _current_engine(self):
        """Return the engine for the current voice/pacing, rebuilding it if either changed."""
        with self._lock:
            backend, voice, speed = self.backend, self.voice, self.speed
        key = (backend, voice, round(float(speed), 3))
        # Held across construction: two threads (a chunk and the preview button) missing the
        # cache would otherwise each load the 325 MB model.
        with self._engine_lock:
            if self._engine is None or self._engine_key != key:
                factory = _KokoroEngine if backend == "kokoro" else _PiperEngine
                started = time.monotonic()
                # Built from the same snapshot as the key, so a concurrent reconfigure can
                # never store a new-voice engine under the old key.
                engine = factory(voice, speed)
                self._engine = engine
                self._engine_key = key
                self._rate = engine.rate
                if time.monotonic() - started > 0.5:
                    print(
                        f"[tts] loaded {voice} at speed {speed} "
                        f"({time.monotonic() - started:.1f}s), applying to the next chunk"
                    )
            return self._engine

    def _neural_synth(self) -> None:
        try:
            self._current_engine()
        except Exception as exc:
            self.error = str(exc)
            self._failed = True
            self._ready.set()
            return
        self._ready.set()

        held: tuple | None = None

        while True:
            # Wait a beat for the next chunk: whatever arrives last ends the reply and
            # gets a proper falling intonation instead of trailing off mid-thought.
            try:
                item = self._sentences.get(
                    timeout=config.NEURAL_FINAL_DEBOUNCE if held else 0.2
                )
            except queue.Empty:
                if held:
                    self._emit(held, final=True)
                    held = None
                continue

            if item is None:
                if held:
                    self._emit(held, final=True)
                return
            if held:
                self._emit(held, final=False)
            held = item

    def _emit(self, item: tuple, final: bool) -> None:
        seq, text = item
        try:
            if seq <= self._skip_seq:
                return
            text = for_speech(text, final=final)
            if not text:
                return
            self._synth_busy.set()
            try:
                engine = self._current_engine()
                with self._synth_lock:
                    audio = engine.synth(text)
                    rate = engine.rate
                if not len(audio) or seq <= self._skip_seq:
                    return
                gap = np.zeros(int(config.NEURAL_GAP_SECONDS * rate), dtype=np.float32)
                audio = np.concatenate([audio, gap])
                if self.sink is not None:
                    self.sink(seq, audio, rate)
                else:
                    self._put(self._clips, (seq, audio, rate))
            except Exception as exc:
                self.error = str(exc)
            finally:
                self._synth_busy.clear()
        finally:
            # Runs on every path, including an interrupted chunk, so pending() can never
            # get stuck counting work that will never happen.
            with self._lock:
                if self._outstanding > 0:
                    self._outstanding -= 1

    def _neural_play(self) -> None:
        self._ready.wait(timeout=120)
        if self.error:
            return
        while True:
            try:
                item = self._clips.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                return
            seq, audio, rate = item
            if seq > self._skip_seq:
                try:
                    self._play(seq, audio, rate)
                except Exception as exc:
                    self.error = str(exc)

    def _play(self, seq: int, audio: np.ndarray, rate: int) -> None:
        self._speaking.set()
        try:
            with sd.OutputStream(samplerate=rate, channels=1, dtype="float32") as stream:
                for start in range(0, len(audio), PLAY_BLOCK):
                    if self._skip_seq >= seq:
                        return
                    stream.write(audio[start : start + PLAY_BLOCK])
        finally:
            self._speaking.clear()

    # ---------- sapi ----------

    def _sapi_loop(self) -> None:
        try:
            self._engine, self._speak_async = self._make_sapi_engine()
        except Exception as exc:
            self.error = str(exc)
            self._failed = True
            self._ready.set()
            return
        self._ready.set()

        while True:
            item = self._next_sentence()
            if item == "empty":
                continue
            if item is None:
                return
            seq, text = item
            if seq <= self._skip_seq:
                continue
            text = for_speech(text, final=True)
            if not text:
                continue
            self._synth_busy.set()
            self._speaking.set()
            try:
                if self._speak_async is None:  # pyttsx3 fallback
                    self._engine.say(text)
                    self._engine.runAndWait()
                else:
                    self._speak_async(text)
                    self._engine.WaitUntilDone(30)
            except Exception as exc:
                self.error = str(exc)
            finally:
                self._speaking.clear()
                self._synth_busy.clear()

    def _make_sapi_engine(self):
        try:
            import win32com.client as win32

            engine = win32.Dispatch("SAPI.SpVoice")
            engine.Rate = max(-10, min(10, self.rate))
            if self.voice:
                for candidate in engine.GetVoices():
                    if self.voice.lower() in candidate.GetDescription().lower():
                        engine.Voice = candidate
                        break
            return engine, engine.Speak
        except ImportError:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            return engine, None

    # ---------- public api ----------

    def say(self, text: str) -> None:
        if self.backend == "silent" or self._failed:
            return
        if not for_speech(text):
            return
        with self._lock:
            self._seq += 1
            seq = self._seq
        self._put_sentence((seq, text))

    def flush(self, on_key=None) -> bool:
        """Wait until speech finishes. Returns False if the user skipped it."""
        while self.pending:
            if on_key and on_key():
                self.skip()
                return False
            time.sleep(0.04)
        return True

    def skip(self) -> None:
        with self._lock:
            self._skip_seq = self._seq
            self._outstanding = 0
        for q in (self._sentences, self._clips):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        if self.backend == "sapi" and self._engine is not None and self._speak_async is not None:
            try:
                self._engine.Speak("", SPV_PURGE)
            except Exception:
                pass
        self._speaking.clear()

    def shutdown(self) -> None:
        self.skip()
        for _ in self._threads:
            self._sentences.put(None)
            self._clips.put(None)
        for thread in self._threads:
            thread.join(timeout=3)
