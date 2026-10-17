"""Speech to text. Local Whisper by default, Google's free endpoint as a fallback."""

import io
import wave

import numpy as np

from . import config


class WhisperSTT:
    """Local, offline transcription via faster-whisper (CTranslate2, no torch)."""

    name = "whisper"

    def __init__(self) -> None:
        self._model = None

    def load(self) -> None:
        if self._model is None:
            from faster_whisper import WhisperModel

            print(
                f"[stt] loading whisper '{config.WHISPER_MODEL}' "
                f"({config.WHISPER_DEVICE}/{config.WHISPER_COMPUTE}), first run downloads it..."
            )
            self._model = WhisperModel(
                config.WHISPER_MODEL,
                device=config.WHISPER_DEVICE,
                compute_type=config.WHISPER_COMPUTE,
            )
            print("[stt] ready")

    def transcribe(self, audio: np.ndarray) -> str:
        self.load()
        segments, _info = self._model.transcribe(
            audio,
            language=config.WHISPER_LANGUAGE or None,
            beam_size=1,
            vad_filter=True,
            temperature=0.0,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


class GoogleSTT:
    """Free, no API key, requires internet. Good for testing the pipeline fast."""

    name = "google"

    def load(self) -> None:
        print("[stt] using Google's free speech endpoint")

    def transcribe(self, audio: np.ndarray) -> str:
        import speech_recognition as sr

        pcm = np.clip(audio, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype("<i2")
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(config.SAMPLE_RATE)
            handle.writeframes(pcm.tobytes())

        recognizer = sr.Recognizer()
        with sr.AudioData(buffer.getvalue(), config.SAMPLE_RATE, 2) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data).strip()


def create(backend: str = config.STT_BACKEND):
    if backend == "google":
        return GoogleSTT()
    if backend == "whisper":
        return WhisperSTT()
    raise ValueError(f"unknown STT backend: {backend}")