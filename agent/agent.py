"""Conversation state and the tool-calling turn loop."""

import json

from . import config, llm, tools


PREAMBLE = (
    "let me",
    "i'll look",
    "i will look",
    "one moment",
    "give me a",
    "searching",
    "checking",
    "looking that up",
    "hold on",
    "just a second",
)


def _is_preamble(text: str) -> bool:
    """True when the text is only the model announcing that it is about to use a tool."""
    lowered = text.lower().strip(" .,")
    if len(lowered) > 80:
        return False
    return any(lowered.startswith(phrase) for phrase in PREAMBLE)


class Agent:
    """Keeps history and turns a transcript into a spoken reply."""

    def __init__(self, model: str = config.LLM_MODEL):
        self.model = model
        self.history: list[dict] = [{"role": "system", "content": config.SYSTEM_PROMPT}]

    def reset(self) -> None:
        self.history = [{"role": "system", "content": config.SYSTEM_PROMPT}]

    def _truncate(self) -> None:
        system, rest = self.history[:1], self.history[1:]
        self.history = system + rest[-2 * config.HISTORY_TURNS :]

    def turn(self, transcript: str):
        """Yield reply text as it streams. History is updated once exhausted.

        With tools enabled the model often writes "let me look that up" *before* calling
        one. That round's text is held back rather than spoken: otherwise the user hears
        the preamble, then the answer, then the preamble again in the history - and the
        voice reads raw search snippets mid-sentence. Only the final round is spoken.
        """
        self.history.append({"role": "user", "content": transcript})
        self._truncate()

        rounds = config.WEB_MAX_ROUNDS if config.WEB_ENABLED else config.MAX_TOOL_ROUNDS
        final = ""
        for _ in range(max(1, rounds)):
            reply = llm.chat(self.history, tools.schemas(), self.model)
            parts: list[str] = []
            for chunk in reply:
                parts.append(chunk)

            if not reply.tool_calls:
                final = "".join(parts).strip()
                if final:
                    for piece in parts:
                        yield piece
                break

            spoken = "".join(parts).strip()
            # Keep the model's own words in history so it can refer back, but only if it
            # said anything that is not just announcing the tool call.
            self.history.append(
                {
                    "role": "assistant",
                    "content": "" if _is_preamble(spoken) else spoken,
                    "tool_calls": reply.tool_calls,
                }
            )
            for call in reply.tool_calls:
                self.history.append({"role": "tool", "content": self._run_tool(call)})
            self._truncate()

        if final:
            self.history.append({"role": "assistant", "content": final})
            self._truncate()

    @staticmethod
    def _run_tool(call: dict) -> str:
        function = call.get("function", {})
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return f"bad arguments: {arguments}"
        return tools.run(function.get("name", ""), arguments)