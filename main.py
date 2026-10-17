"""A small local voice agent: talk -> listen -> think -> speak.

    python main.py            # push-to-talk with the default settings
    python main.py --help     # all options
"""

import argparse
import sys
import threading

from agent import audio, config, llm, stt
from agent.agent import Agent
from agent.tts import Speaker, for_speech, stream_chunks

BANNER = r"""
  __      __  _              _             _
   \ \    / /__| |_ __ _  ___(_) ___ __ _  | |_ ___
    \ \/\/ / -_)  _/ _` |/ _ \ | (_-<  _` |  _/ _ \
     \_/\_/\___|\__\__,_/\___/_|_|__/\__,_|\__\___/
        local voice agent  -  whisper + ollama + offline tts
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local push-to-talk voice agent")
    parser.add_argument("--model", default=config.LLM_MODEL, help="Ollama model name")
    parser.add_argument("--stt", default=config.STT_BACKEND, choices=["whisper", "google"])
    parser.add_argument("--tts", default=config.TTS_BACKEND,
                        choices=["kokoro", "piper", "sapi", "silent"])
    parser.add_argument("--whisper-model", default=config.WHISPER_MODEL)
    parser.add_argument("--language", default=config.WHISPER_LANGUAGE or "en")
    parser.add_argument("--rate", type=int, default=config.TTS_RATE,
                        help="sapi only: -10 (slow) to 10 (fast)")
    parser.add_argument("--voice", default=None,
                        help="kokoro voice (af_heart), piper voice (en_US-amy-medium), "
                             "or sapi voice substring")
    parser.add_argument("--speed", type=float, default=config.NEURAL_LENGTH_SCALE,
                        help="neural pacing, 1.0 normal, 1.1 slower, 0.9 faster")
    parser.add_argument("--no-memory", action="store_true", help="forget context each turn")
    parser.add_argument("--say", metavar="TEXT",
                        help="speak TEXT through the voice pipeline and exit")
    parser.add_argument("--list-devices", action="store_true", help="show audio devices and exit")
    return parser.parse_args()


def apply_args(args: argparse.Namespace) -> None:
    config.LLM_MODEL = args.model
    config.STT_BACKEND = args.stt
    config.TTS_BACKEND = args.tts
    config.WHISPER_MODEL = args.whisper_model
    config.WHISPER_LANGUAGE = args.language
    config.TTS_RATE = args.rate
    config.NEURAL_LENGTH_SCALE = args.speed
    if args.voice is not None:
        if args.tts == "kokoro":
            config.KOKORO_VOICE = args.voice
        elif args.tts == "piper":
            config.PIPER_VOICE = args.voice
        else:
            config.TTS_VOICE = args.voice
    if args.no_memory:
        config.HISTORY_TURNS = 1


def check_backend(args: argparse.Namespace) -> bool:
    if not llm.is_available():
        print(f"[error] no Ollama server at {config.OLLAMA_HOST}")
        print("        1. install Ollama:  https://ollama.com/download")
        print(f"        2. pull the model:  ollama pull {args.model}")
        return False
    if args.model not in llm.installed_models():
        print(f"[warn] '{args.model}' is not pulled yet ->  ollama pull {args.model}")
    return True


def run_turn(agent: Agent, listener, speaker: Speaker) -> None:
    print("you  > ", end="")
    clip = audio.record_until_silence(
        on_tick=lambda level, threshold, recording: audio.draw_meter(level, threshold, recording)
    )
    if clip.size == 0:
        print("        (didn't hear anything)\n")
        return

    try:
        transcript = listener.transcribe(clip)
    except Exception as exc:
        print(f"[stt error] {exc}\n")
        return
    if not transcript:
        print("        (nothing intelligible)\n")
        return

    print(f"\nyou  > {transcript}")
    print("bot > ", end="")

    chunks = agent.turn(transcript)
    for chunk in stream_chunks(chunks):
        speaker.say(chunk)

    print()
    if speaker.backend != "silent":
        speaker.flush(on_key=_key_pressed)
    print()


def build_speaker(args: argparse.Namespace) -> Speaker:
    speaker = Speaker(args.tts, args.rate, config.TTS_VOICE, args.speed)
    neural = args.tts in ("kokoro", "piper")
    if neural:
        print(f"[tts] loading neural voice '{speaker.voice}'...")
    speaker.start()
    if speaker.error:
        print(f"[warn] tts unavailable ({speaker.error}); continuing in text mode")
        speaker.backend = "silent"
    elif neural:
        print("[tts] voice ready")
    return speaker


def main() -> int:
    args = parse_args()
    if args.list_devices:
        for index, name in audio.list_devices():
            print(f"{index}: {name}")
        return 0

    apply_args(args)
    print(BANNER)

    if args.say:
        speaker = build_speaker(args)
        if speaker.error:
            return 1
        chunks = list(stream_chunks(iter([args.say])))
        for i, chunk in enumerate(chunks):
            final = i == len(chunks) - 1
            print(f"  chunk {i + 1}/{len(chunks)} [{'final' if final else 'continuing'}]: "
                  f"{for_speech(chunk, final=final)}")
            speaker.say(chunk)
        speaker.flush()
        speaker.shutdown()
        return 0

    if not check_backend(args):
        return 1

    listener = stt.create(args.stt)
    threading.Thread(target=listener.load, daemon=True).start()

    speaker = build_speaker(args)

    threading.Thread(target=llm.warm, args=(args.model,), daemon=True).start()
    agent = Agent(args.model)
    voice = speaker.voice if args.tts in ("kokoro", "piper") else args.model
    print(f"\nbrain: {args.model} ({voice})   |   warming up, talk whenever")
    print("ENTER = talk   |   ENTER again = stop early   |   type q = quit   |   r = reset memory\n")

    try:
        while True:
            try:
                command = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            lowered = command.lower()
            if lowered in ("q", "quit", "exit"):
                break
            if lowered in ("r", "reset"):
                agent.reset()
                print("[memory cleared]\n")
                continue
            if lowered.startswith("w "):
                save_clip(command[2:].strip())
                continue

            run_turn(agent, listener, speaker)
    finally:
        speaker.shutdown()

    print("bye.")
    return 0


def save_clip(path: str) -> None:
    clip = audio.record_until_key(lambda: _key_pressed())
    if clip is None or clip.size == 0:
        print("(nothing recorded)")
        return
    audio.save_wav(path, clip)
    print(f"[saved {path}]")


def _key_pressed() -> bool:
    import msvcrt

    while msvcrt.kbhit():
        if msvcrt.getwch() == "\r":
            return True
    return False


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()