"""Microphone capture with adaptive, silence-based endpointing."""

import sys
import time
from collections import deque

import numpy as np
import sounddevice as sd

from . import config

CHUNK_SECONDS = config.CHUNK / config.SAMPLE_RATE
BLOCKS_PER_SECOND = 1.0 / CHUNK_SECONDS


def rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(block))) + 1e-12)


def draw_meter(level: float, threshold: float, recording: bool) -> None:
    width = 24
    filled = min(width, int(level / max(threshold, 1e-6) * width))
    bar = "#" * filled + "." * (width - filled)
    label = "REC" if recording else "..."
    sys.stdout.write(f"\r   {label} |{bar}| {level:5.3f}")
    sys.stdout.flush()


def _clear_line() -> None:
    sys.stdout.write("\r" + " " * 64 + "\r")
    sys.stdout.flush()


def _threshold(levels: "deque[float]") -> float:
    """Track the noise floor: a low percentile of recent levels, times a margin."""
    if not levels:
        return config.MIN_THRESHOLD
    floor = float(np.percentile(np.fromiter(levels, dtype=np.float32), 25))
    return max(config.MIN_THRESHOLD, floor * config.NOISE_MARGIN)


def record_until_silence(
    on_tick=None,
    max_seconds: float = config.MAX_RECORD_SECONDS,
    silence_seconds: float = config.SILENCE_SECONDS_TO_STOP,
    onset_seconds: float = config.ONSET_SECONDS,
    should_stop=None,
) -> np.ndarray:
    """Record until the speaker stops talking. Empty array if nobody spoke.

    should_stop is polled between blocks: it lets push-to-talk release end the recording
    (keeping what was captured) and lets a disconnect free the microphone at once instead
    of holding it until MAX_RECORD_SECONDS.
    """
    levels: deque[float] = deque(maxlen=int(4 * BLOCKS_PER_SECOND))
    recent: deque[np.ndarray] = deque(maxlen=int(0.4 * BLOCKS_PER_SECOND))
    frames: list[np.ndarray] = []
    speaking = False
    onset_run = 0.0
    quiet_run = 0.0
    threshold = config.MIN_THRESHOLD
    max_blocks = int(max_seconds / CHUNK_SECONDS)
    quiet_blocks = int(silence_seconds / CHUNK_SECONDS)

    with sd.InputStream(
        samplerate=config.SAMPLE_RATE, channels=1, dtype="float32", blocksize=config.CHUNK
    ) as stream:
        for _ in range(max_blocks):
            if should_stop is not None and should_stop():
                break
            block, _ = stream.read(config.CHUNK)
            flat = block.reshape(-1).copy()
            level = rms(flat)

            if speaking:
                frames.append(flat)
                quiet_run = quiet_run + CHUNK_SECONDS if level <= threshold else 0.0
                if on_tick:
                    on_tick(level, threshold, True)
                if quiet_run >= silence_seconds:
                    break
                continue

            levels.append(level)
            recent.append(flat)
            threshold = _threshold(levels)
            onset_run = onset_run + CHUNK_SECONDS if level > threshold else 0.0

            if onset_run >= onset_seconds:
                speaking = True
                frames.extend(recent)
                recent.clear()
                onset_run = 0.0
                quiet_run = 0.0
            elif on_tick:
                on_tick(level, threshold, False)

    _clear_line()
    if not frames:
        return np.zeros(0, dtype=np.float32)

    audio = np.concatenate(frames)
    audio = _trim_trailing_silence(audio, threshold)
    if len(audio) / config.SAMPLE_RATE < config.MIN_SPEECH_SECONDS:
        return np.zeros(0, dtype=np.float32)
    return audio


def _trim_trailing_silence(audio: np.ndarray, threshold: float, keep: float = 0.3) -> np.ndarray:
    """Drop the dead air at the end so the transcriber gets a tight clip."""
    blocks = len(audio) // config.CHUNK
    if blocks < 2:
        return audio
    tail: list[int] = []
    for index in range(blocks - 1, -1, -1):
        block = audio[index * config.CHUNK : (index + 1) * config.CHUNK]
        if rms(block) > threshold:
            break
        tail.append(index)
        if len(tail) >= int(keep / CHUNK_SECONDS):
            break
    if tail:
        end = min(tail) * config.CHUNK
        audio = audio[:end]
    return audio


def record_until_key(is_pressed, max_seconds: float = 60.0) -> np.ndarray:
    """Record until `is_pressed()` returns True (e.g. an Enter keypress)."""
    frames: list[np.ndarray] = []
    deadline = time.time() + max_seconds
    with sd.InputStream(
        samplerate=config.SAMPLE_RATE, channels=1, dtype="float32", blocksize=config.CHUNK
    ) as stream:
        while time.time() < deadline:
            block, _ = stream.read(config.CHUNK)
            frames.append(block.reshape(-1).copy())
            if is_pressed():
                break
    _clear_line()
    return np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)


def save_wav(path: str, audio: np.ndarray, sample_rate: int = config.SAMPLE_RATE) -> None:
    import wave

    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def list_devices() -> list[tuple[int, str]]:
    return [(i, d["name"]) for i, d in enumerate(sd.query_devices())]