"""Safety, ambiguity and incomplete-query checks (ported from the notebook, with bug fixes)."""
import re

from . import config

BLOCKED_PATTERNS = [
    r"ignore (all )?(the )?previous instructions",
    r"reveal (the |your )?system prompt",
    r"bypass (the )?safety",
    r"(give|show|tell) me (the |your )?api key",
    r"\bjailbreak\b",
    r"\bhack(ing|er|ers)?\b",
    r"\bmalware\b",
]
_BLOCKED = [re.compile(p) for p in BLOCKED_PATTERNS]

AMBIGUOUS_PHRASES = [
    "explain that algorithm",
    "explain this algorithm",
    "explain that model",
    "explain this model",
    "tell me more about it",
    "explain it",
    "what is it",
    "how does it work",
    "that method",
    "this method",
]
_AMBIGUOUS = [re.compile(rf"\b{re.escape(p)}\b") for p in AMBIGUOUS_PHRASES]

INCOMPLETE_ENDINGS = ["and", "or", "between", "difference between", "explain", "what is", "how to", "why is"]

UNSAFE_MESSAGE = (
    "This request may be unsafe or attempts to bypass chatbot rules. "
    "Please ask a learning-related question."
)
INCOMPLETE_MESSAGE = (
    "Your question seems incomplete. Please provide more details so I can answer correctly."
)
AMBIGUOUS_MESSAGE = (
    "Your question is unclear. Please specify the AI or Machine Learning "
    "concept, algorithm, or model you want me to explain."
)


def _words(query: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", query.lower())


def safety_check(query: str) -> bool:
    """True if the query is safe."""
    q = query.lower()
    return not any(p.search(q) for p in _BLOCKED)


def is_incomplete_query(query: str) -> bool:
    # Word-level match, so "understand" does not count as ending with "and".
    text = " ".join(_words(query))
    return any(text == e or text.endswith(" " + e) for e in INCOMPLETE_ENDINGS)


def is_ambiguous_query(query: str) -> bool:
    words = _words(query)
    text = " ".join(words)
    if any(p.search(text) for p in _AMBIGUOUS):
        return True
    return len(words) < config.MIN_QUERY_WORDS


def check_query(query: str) -> str | None:
    """Return a canned reply if the query should not go to the LLM, else None."""
    if not safety_check(query):
        return UNSAFE_MESSAGE
    if is_incomplete_query(query):
        return INCOMPLETE_MESSAGE
    if is_ambiguous_query(query):
        return AMBIGUOUS_MESSAGE
    return None
