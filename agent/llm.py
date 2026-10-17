"""Minimal streaming chat client for a local Ollama server (stdlib only)."""

import json
import urllib.request

from . import config


class OllamaError(RuntimeError):
    pass


def is_available() -> bool:
    try:
        with urllib.request.urlopen(f"{config.OLLAMA_HOST}/api/tags", timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def installed_models() -> list[str]:
    try:
        with urllib.request.urlopen(f"{config.OLLAMA_HOST}/api/tags", timeout=5) as response:
            payload = json.load(response)
        return [m["name"] for m in payload.get("models", [])]
    except Exception:
        return []


def _post(messages, tools, model, stream):
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "tools": tools or [],
            "stream": stream,
            "keep_alive": config.LLM_KEEP_ALIVE,
            "options": {
                "temperature": config.LLM_TEMPERATURE,
                "num_predict": config.LLM_MAX_TOKENS,
                "num_ctx": config.LLM_NUM_CTX,
            },
        }
    ).encode()
    request = urllib.request.Request(
        f"{config.OLLAMA_HOST}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=180)


class Reply:
    """Iterate over text chunks; `text` and `tool_calls` are valid once exhausted."""

    def __init__(self, messages, tools=None, model=None):
        self._messages = messages
        self._tools = tools
        self._model = model or config.LLM_MODEL
        self.text = ""
        self.tool_calls = None

    def __iter__(self):
        chunks: list[str] = []
        with _post(self._messages, self._tools, self._model, True) as response:
            for line in response:
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                if event.get("error"):
                    raise OllamaError(event["error"])
                message = event.get("message") or {}
                piece = message.get("content") or ""
                if piece:
                    chunks.append(piece)
                    yield piece
                if message.get("tool_calls"):
                    self.tool_calls = message["tool_calls"]
                if event.get("done"):
                    break
        self.text = "".join(chunks).strip()


def chat(messages, tools=None, model=None) -> Reply:
    return Reply(messages, tools, model)


def warm(model=None) -> bool:
    """Load the model into memory so the first real reply is not slowed by it."""
    model = model or config.LLM_MODEL
    body = json.dumps(
        {
            "model": model,
            "prompt": "",
            "stream": False,
            "keep_alive": config.LLM_KEEP_ALIVE,
            "options": {"num_predict": 1},
        }
    ).encode()
    request = urllib.request.Request(
        f"{config.OLLAMA_HOST}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=300):
            return True
    except Exception:
        return False