"""Small, safe tools the model can call.

Everything here is read-only and offline by default except the `web_*` tools, which are
opt-in behind config.WEB_ENABLED. There is no code execution: `calculate` walks a tiny
whitelist AST instead of eval().
"""

import ast
import datetime as dt
import html
import operator
import re
import urllib.parse
import urllib.request

from . import config

# DuckDuckGo's HTML endpoint returns an empty result page to anything that does not look
# like a browser: the custom agent string above got zero results, a browser UA gets ten.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
MAX_FETCH_BYTES = 400_000

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _evaluate(node):
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError("only numbers allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](_evaluate(node.operand))
    raise ValueError("unsupported expression")


def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. '19.99 * 1.2 + 3'."""
    if len(expression) > 200:
        return "expression too long"
    try:
        value = _evaluate(ast.parse(expression, mode="eval"))
        if isinstance(value, complex):
            return str(value)
        return f"{value:.6g}"
    except ZeroDivisionError:
        return "division by zero"
    except Exception:
        return "could not evaluate that expression"


def current_time() -> str:
    """Return the current local date and time."""
    now = dt.datetime.now().astimezone()
    return now.strftime("%A, %d %B %Y at %I:%M %p %Z").strip()


# ---------- internet access (opt-in) ----------


def _get(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read(MAX_FETCH_BYTES)


def web_search(query: str, results: int = 5) -> str:
    """Search the web and return the top results as title, url and snippet."""
    if not config.WEB_ENABLED:
        return "internet access is switched off"
    query = (query or "").strip()
    if not query:
        return "no search query given"
    results = max(1, min(10, int(results or 5)))
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    try:
        page = _get(url, config.WEB_TIMEOUT).decode("utf-8", "replace")
    except Exception as exc:
        return f"the search failed: {exc}"

    # DuckDuckGo's HTML endpoint: results are anchors with class result__a, each followed
    # by a result__snippet div.
    blocks = re.findall(
        r'<a[^>]+class="result__a"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>'
        r'(?:.*?class="result__snippet"[^>]*>(?P<snippet>.*?)</a>)?',
        page,
        flags=re.S,
    )
    if not blocks:
        return "no search results came back"
    lines = []
    for index, (link, title, snippet) in enumerate(blocks[:results], start=1):
        clean_title = _text(title)
        if link.startswith("//duckduckgo.com/l/?uddg="):
            link = urllib.parse.unquote(link.split("uddg=", 1)[1].split("&", 1)[0])
        lines.append(f"{index}. {clean_title}\n{link}\n{_text(snippet)[:220]}")
    return "\n".join(lines)


def web_fetch(url: str) -> str:
    """Download a web page and return its readable text."""
    if not config.WEB_ENABLED:
        return "internet access is switched off"
    url = (url or "").strip()
    if not url:
        return "no url given"
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        raw = _get(url, config.WEB_TIMEOUT).decode("utf-8", "replace")
    except Exception as exc:
        return f"could not load that page: {exc}"
    return _page_text(raw)[:config.WEB_MAX_CHARS]


def _text(fragment: str) -> str:
    fragment = re.sub(r"<[^>]+>", " ", fragment)
    return re.sub(r"\s+", " ", html.unescape(fragment)).strip()


def _page_text(page: str) -> str:
    """Crude readability: the point is facts for the model, not a layout engine."""
    page = re.sub(r"(?is)<(script|style|nav|header|footer|noscript|svg)[^>]*>.*?</\1>", " ", page)
    title = re.search(r"(?is)<title[^>]*>(.*?)</title>", page)
    body = _text(page)
    lead = f"{_text(title.group(1))}\n" if title else ""
    return (lead + body).strip()


def _schema(name: str, description: str, properties: dict, required: list) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


SCHEMAS = [
    _schema("current_time", "Get the current local date and time.", {}, []),
    _schema(
        "calculate",
        "Evaluate an arithmetic expression.",
        {"expression": {"type": "string"}},
        ["expression"],
    ),
    _schema(
        "web_search",
        "Search the internet and get titles, urls and snippets. Use it for current events, "
        "prices, weather, or anything that might have changed recently.",
        {
            "query": {"type": "string", "description": "what to search for"},
            "results": {"type": "integer", "description": "how many results (1-10)"},
        },
        ["query"],
    ),
    _schema(
        "web_fetch",
        "Download a web page and read its text. Use it after web_search to read a result, or "
        "to read a url the user mentions.",
        {"url": {"type": "string", "description": "the page to read"}},
        ["url"],
    ),
]

HANDLERS = {
    "current_time": current_time,
    "calculate": calculate,
    "web_search": web_search,
    "web_fetch": web_fetch,
}

def schemas() -> list[dict]:
    """Tools the model may call. Web tools are withheld when they cannot work.

    Offering a tool that always answers "switched off" wastes a round trip on every
    factual question and invites the model to report a fabricated failure.
    """
    if config.WEB_ENABLED:
        return list(SCHEMAS)
    return [item for item in SCHEMAS if item["function"]["name"] not in ("web_search", "web_fetch")]


def run(name: str, arguments: dict) -> str:
    handler = HANDLERS.get(name)
    if handler is None:
        return f"unknown tool: {name}"
    try:
        return str(handler(**arguments))
    except TypeError as exc:
        return f"bad arguments: {exc}"