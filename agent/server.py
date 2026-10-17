"""Web UI backend for the voice agent.

Standard library only: a threaded HTTP server that serves the dashboard, takes microphone
audio from the browser as raw PCM, and streams the reply back as text plus audio events.
The browser is the speaker here, so the Speaker gets a sink instead of a playback thread.
"""

import base64
import io
import json
import mimetypes
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from . import config, llm, stt
from .agent import Agent
from .tts import Speaker, kokoro_paths, stream_chunks

WEB_ROOT = Path(__file__).resolve().parent.parent / "web"
HOST = "127.0.0.1"
PORT = 8765


def wav_bytes(audio: np.ndarray, rate: int) -> bytes:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


class Hub:
    """Shared state: one speaker, one listener per model, one turn at a time."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.agents: dict[str, Agent] = {}
        self.listener = stt.create(config.STT_BACKEND)
        self.playback_done = threading.Event()
        self.playback_done.set()
        self.active_turn: "Turn | None" = None
        self.turn_seq = 0
        self.skip_seq = 0
        self.log_lock = threading.Lock()
        self.client_log: list[dict] = []
        self._models: list[str] = []
        self._models_at = 0.0
        self._voices: list[str] = []
        self.speaker = Speaker(
            config.TTS_BACKEND, config.TTS_RATE, "", config.NEURAL_LENGTH_SCALE, sink=self.on_clip
        )

    def start(self) -> None:
        threading.Thread(target=self.listener.load, daemon=True).start()
        threading.Thread(target=llm.warm, args=(config.LLM_MODEL,), daemon=True).start()
        self.speaker.start()
        if self.speaker.error:
            print(f"[warn] tts unavailable ({self.speaker.error})")

    def agent(self, model: str) -> Agent:
        with self.lock:
            agent = self.agents.get(model)
            if agent is None:
                agent = self.agents[model] = Agent(model)
                threading.Thread(target=llm.warm, args=(model,), daemon=True).start()
            return agent

    def current_model(self) -> str:
        return config.LLM_MODEL

    def installed_models(self) -> list[str]:
        """Ollama's model list takes ~2s, so it is cached briefly.

        The dashboard awaits this on every slider release, which made each adjustment feel
        like a two second freeze.
        """
        now = time.monotonic()
        with self.lock:
            if self._models and now - self._models_at < 30:
                return self._models
        models = llm.installed_models()
        with self.lock:
            self._models = models
            self._models_at = now
            return models

    def voices(self) -> list[str]:
        """Cached: the Kokoro voice pack is a 28 MB npz and never changes while running."""
        with self.lock:
            if self._voices:
                return self._voices
        try:
            with np.load(kokoro_paths()[1]) as pack:
                voices = sorted(pack.files)
        except Exception:
            return []
        with self.lock:
            self._voices = voices
        return voices

    # ---------- turn plumbing ----------

    def on_clip(self, seq: int, audio: np.ndarray, rate: int) -> None:
        """Called from the synthesis thread as each chunk is ready."""
        turn = self.active_turn
        # min_seq keeps a chunk from an abandoned turn out of the next one. Without it, the
        # leftover audio of an interrupted reply was spoken during the following reply.
        if turn is not None and seq > self.skip_seq and seq >= turn.min_seq:
            turn.send_audio(audio, rate)

    def begin_turn(self) -> "Turn | None":
        """Start a turn, or refuse it if one is already in flight."""
        with self.lock:
            if self.active_turn is not None and not self.active_turn.finished.is_set():
                return None
            self.turn_seq += 1
            turn = Turn(self.turn_seq, self.speaker.seq + 1)
            self.active_turn = turn
            self.playback_done.clear()
            return turn

    def end_turn(self, turn: "Turn") -> None:
        with self.lock:
            if self.active_turn is turn:
                self.active_turn = None
        self.playback_done.set()

    def interrupt(self) -> None:
        self.skip_seq = self.speaker._seq
        self.speaker.skip()
        self.stop_listening()
        if self.active_turn is not None:
            self.active_turn.cancelled = True
        self.playback_done.set()

    def stop_listening(self) -> None:
        """End an app-side recording but keep the turn alive (push-to-talk release)."""
        with self.lock:
            turn = self.active_turn
        if turn is not None:
            turn.stop_listening = True

    def finish_playback(self, turn_id: int) -> None:
        with self.lock:
            turn = self.active_turn
        if turn is not None and turn.id == turn_id:
            self.playback_done.set()


class Turn:
    """One question/answer exchange, streamed to the browser over SSE."""

    # Audio is bulk data, so a dead client's queue is trimmed rather than grown without
    # bound while a 30 s recording finishes.
    MAX_AUDIO_EVENTS = 64

    def __init__(self, turn_id: int, min_seq: int = 0) -> None:
        self.id = turn_id
        self.min_seq = min_seq
        self.cancelled = False
        self.stop_listening = False
        self.queue: list[dict] = []
        self.event = threading.Event()
        self.transcript = ""
        self.reply = ""
        self.timings: dict[str, float] = {}
        self.sent_audio = False
        self.finished = threading.Event()
        self.started = time.time()

    def push(self, event: dict) -> None:
        event["data"]["turn"] = self.id
        if event["event"] == "audio":
            kept = [item for item in self.queue if item["event"] != "audio"]
            if len(kept) < self.MAX_AUDIO_EVENTS:
                self.queue[:] = kept
            else:
                self.queue[:] = kept[-(self.MAX_AUDIO_EVENTS - 1) :]
        self.queue.append(event)
        self.event.set()

    def send(self, name: str, **data) -> None:
        self.push({"event": name, "data": data})

    def send_audio(self, audio: np.ndarray, rate: int) -> None:
        self.sent_audio = True
        payload = base64.b64encode(wav_bytes(audio, rate)).decode("ascii")
        self.send("audio", rate=rate, wav=payload)

    def status(self, phase: str, **extra) -> None:
        self.send("status", phase=phase, **extra)


def sse(event: dict) -> bytes:
    return f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n".encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    hub: Hub = None  # set by serve()
    protocol_version = "HTTP/1.1"
    _print_lock = threading.Lock()

    def log_message(self, fmt, *args) -> None:
        pass

    def note(self, note: str) -> None:
        # Locked: ThreadingHTTPServer runs one thread per connection and interleaved
        # partial lines make the access log useless.
        with Handler._print_lock:
            print(f"[http] {self.command} {self.path} {note}", flush=True)

    # ---------- helpers ----------

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, code: int = 200) -> None:
        self._send(code, json.dumps(data).encode("utf-8"), "application/json")

    def _read_json(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}
        return body if isinstance(body, dict) else {}

    def _read_pcm(self) -> np.ndarray:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length else b""
        if len(raw) % 2:
            raw = raw[:-1]
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0

    def _drain(self) -> None:
        """Consume an unread request body.

        The connection is kept alive, so a route that answers without reading the body
        leaves the next request on the same socket to be parsed starting in the middle of
        the old payload - the browser then gets a 501 and the turn silently does nothing.
        """
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length:
            self.rfile.read(length)

    # ---------- routes ----------

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            path = "/index.html"
        if path == "/api/state":
            return self._json(self.hub_state())
        if path == "/api/client-log":
            with self.hub.log_lock:
                return self._json({"entries": list(self.hub.client_log)})
        target = (WEB_ROOT / path.lstrip("/")).resolve()
        if WEB_ROOT in target.parents and target.is_file():
            kind = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            return self._send(200, target.read_bytes(), kind)
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/turn":
            self.note("stream")
            return self._stream_turn()
        if path == "/api/listen":
            self.note("stream")
            return self._stream_listen()
        if path == "/api/settings":
            body = self._read_json()
            return self._json(self.apply_settings(body))
        if path == "/api/preview":
            # Pass the body in: _preview reads it itself, and reading it here as well would
            # block on a socket whose only body has already been consumed.
            return self._preview(self._read_json())
        if path == "/api/interrupt":
            self._read_json()
            self.hub.interrupt()
            self.note("200")
            return self._json({"ok": True})
        if path == "/api/stop-listening":
            self._read_json()
            self.hub.stop_listening()
            self.note("200")
            return self._json({"ok": True})
        if path == "/api/playback-done":
            body = self._read_json()
            self.hub.finish_playback(int(body.get("turn", 0) or 0))
            self.note("200")
            return self._json({"ok": True})
        if path == "/api/client-log":
            body = self._read_json()
            entry = {"level": body.get("level", "info"), "msg": body.get("msg", "")}
            with self.hub.log_lock:
                self.hub.client_log.append(entry)
                del self.hub.client_log[:-60]
            if entry["level"] in ("error", "warn"):
                with Handler._print_lock:
                    print(f"[browser:{entry['level']}] {entry['msg']}", flush=True)
            return self._json({"ok": True})
        if path == "/api/reset":
            body = self._read_json()
            agent = self.hub.agent(body.get("model") or self.hub.current_model())
            agent.reset()
            self.note("200")
            return self._json({"ok": True})
        self._drain()
        self.note("404")
        self._send(404, b"not found", "text/plain")

    # ---------- api implementations ----------

    def hub_state(self) -> dict:
        speaker = self.hub.speaker
        return {
            "model": self.hub.current_model(),
            "models": self.hub.installed_models(),
            "tts": speaker.backend,
            "voice": speaker.voice,
            "voices": self.hub.voices(),
            "speed": speaker.speed,
            "min_words": config.NEURAL_MIN_CHUNK_WORDS,
            "debounce": config.NEURAL_FINAL_DEBOUNCE,
            "gap": config.NEURAL_GAP_SECONDS,
            "stt": config.STT_BACKEND,
            "whisper_model": config.WHISPER_MODEL,
            "error": speaker.error,
        }

    def apply_settings(self, body: dict) -> dict:
        voice = body.get("voice")
        speed = body.get("speed")
        model = body.get("model")
        known_voices = self.hub.voices()
        if voice and known_voices and voice not in known_voices:
            # Answering 200 for an unknown voice left the agent permanently broken:
            # every later synthesis failed and the error stuck in the header.
            return self._json({"error": f"unknown voice '{voice}'"}, 400)
        self.hub.speaker.reconfigure(voice=voice, speed=speed)
        if speed:
            config.NEURAL_LENGTH_SCALE = float(speed)
        for key, cast in (("min_words", int), ("debounce", float), ("gap", float)):
            if body.get(key) is not None:
                setattr(config, {
                    "min_words": "NEURAL_MIN_CHUNK_WORDS",
                    "debounce": "NEURAL_FINAL_DEBOUNCE",
                    "gap": "NEURAL_GAP_SECONDS",
                }[key], cast(body[key]))
        if model and model != config.LLM_MODEL:
            config.LLM_MODEL = model
            self.hub.agent(model)
        # A rejected value is worth reporting loudly once, but a recovered engine must not
        # leave the message up forever.
        self.hub.speaker.error = None
        return self.hub_state()

    def _preview(self, body: dict) -> None:
        """Synthesize one sample line with the live settings and return a wav."""
        text = (body.get("text") or "Here is how I sound with the current settings.").strip()
        try:
            audio, rate = self.hub.speaker.preview(text)
        except Exception as exc:
            return self._json({"error": str(exc)}, 500)
        self._send(200, wav_bytes(audio, rate), "audio/wav")

    def _stream_turn(self) -> None:
        pcm = self._read_pcm()
        self._drive(pcm)

    def _stream_listen(self) -> None:
        """Record from the app's own microphone, so no browser permission is needed."""
        self._read_json()
        self._drive(None)

    def _open_stream(self, turn: Turn) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()

    def _drive(self, pcm) -> None:
        turn = self.hub.begin_turn()
        if turn is None:
            # One reply at a time: a second concurrent client would have its audio routed
            # to whichever tab happened to hold the active turn.
            return self._json({"error": "a reply is already in progress"}, 409)
        self._open_stream(turn)
        done = threading.Event()

        def capture() -> None:
            """Runs the agent in the background so this thread can stream events out."""
            heard = pcm
            try:
                if heard is None:
                    turn.status("listening")
                    from . import audio

                    heard = audio.record_until_silence(
                        on_tick=lambda level, threshold, recording:
                        turn.send("level", level=round(level, 4), live=bool(recording)),
                        should_stop=lambda: turn.stop_listening or turn.cancelled,
                    )
                    if not len(heard):
                        turn.send("done", transcript="", reply="", timings=turn.timings,
                                  turn=turn.id, reason="no speech")
                        return

                started = time.time()
                turn.status("transcribing")
                transcript = self.hub.listener.transcribe(heard)
                turn.transcript = transcript
                turn.timings["stt"] = time.time() - started
                turn.send("transcript", text=transcript)
                if not transcript:
                    turn.send("done", transcript="", reply="", timings=turn.timings,
                              turn=turn.id, reason="empty")
                    return

                agent = self.hub.agent(self.hub.current_model())
                started = time.time()
                first = None
                turn.status("thinking")
                for chunk in stream_chunks(agent.turn(transcript)):
                    if turn.cancelled:
                        break
                    if first is None:
                        first = time.time() - started
                        turn.timings["llm_first"] = first
                        turn.status("speaking")
                    turn.reply += (" " if turn.reply else "") + chunk
                    turn.send("text", delta=chunk)
                    self.hub.speaker.say(chunk)
                turn.timings["llm"] = time.time() - started

                deadline = time.time() + 120
                while self.hub.speaker.pending and time.time() < deadline and not turn.cancelled:
                    time.sleep(0.05)
                if turn.sent_audio and not turn.cancelled:
                    self.hub.playback_done.wait(timeout=30)
                turn.timings["total"] = time.time() - turn.started
                turn.send("done", transcript=turn.transcript, reply=turn.reply,
                          timings=turn.timings, turn=turn.id,
                          reason="cancelled" if turn.cancelled else "ok")
            except Exception as exc:
                turn.send("done", transcript=turn.transcript, reply=turn.reply,
                          timings=turn.timings, turn=turn.id, error=str(exc))
            finally:
                turn.finished.set()
                # Wakes a disconnected capture thread: record_until_silence polls this
                # between blocks, so it stops holding the microphone almost immediately.
                turn.event.set()
                done.set()

        threading.Thread(target=capture, daemon=True).start()

        while True:
            if not turn.event.wait(timeout=0.5):
                if done.is_set():
                    break
                try:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    turn.cancelled = True
                    break
                continue
            turn.event.clear()
            while turn.queue:
                event = turn.queue.pop(0)
                if event["event"] == "done":
                    done.set()
                try:
                    self.wfile.write(sse(event))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    turn.cancelled = True
                    done.set()
                    break
            if done.is_set() and not turn.queue:
                break

        self.hub.end_turn(turn)
        self.close_connection = True


def serve(host: str = HOST, port: int = PORT) -> None:
    hub = Hub()
    Handler.hub = hub
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    httpd.handle_error = lambda *args: None  # browsers abort streams on reload
    hub.start()
    print(f"[ui] http://{host}:{port}  (ctrl-c to stop)")
    print(f"[ui] brain: {config.LLM_MODEL}   voice: {hub.speaker.voice}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[ui] stopped")
    finally:
        hub.speaker.shutdown()
        httpd.server_close()


if __name__ == "__main__":
    serve()
