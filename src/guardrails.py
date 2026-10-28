import re

INJECTION_MARKERS = ["ignore your instructions", "system prompt", "developer message", "reveal your key"]
PROHIBITED_REPLY_PATTERNS = [
    r"\b\₹?\s*\d+[\d,]*\b",
    r"cash compensation",
    r"full refund will be issued",
    r"we admit fault",
    r"we are to blame",
]


def redact_text(text: str) -> str:
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[EMAIL]", text)
    text = re.sub(r"(?<!\d)(?:\+?\d[\d\s-]{8,}\d)(?!\d)", "[PHONE]", text)
    return text[:4000]


def contains_injection(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in INJECTION_MARKERS)


def validate_reply(reply: str) -> list[str]:
    warnings = []
    if len(reply.split()) > 60:
        warnings.append("Reply exceeds the 60-word policy limit.")
    if sum(reply.lower().count(word) for word in ["sorry", "apologize", "apologies"]) > 1:
        warnings.append("Reply apologizes more than once.")
    for pattern in PROHIBITED_REPLY_PATTERNS:
        if re.search(pattern, reply, flags=re.IGNORECASE):
            warnings.append("Reply contains a prohibited promise or admission.")
            break
    if re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+|\+?\d[\d\s-]{8,}\d", reply):
        warnings.append("Reply may expose contact details.")
    return warnings

