import re

_HORIZONTAL = re.compile(r"[^\S\n]+")
_EXCESS_NEWLINES = re.compile(r"\n{3,}")
_SPACES_BEFORE_NEWLINE = re.compile(r" +\n")

def normalize_text(text: str) -> str:
    """Normalize whitespace in plain extracted text, preserving line breaks."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HORIZONTAL.sub(" ", text)
    text = _SPACES_BEFORE_NEWLINE.sub("\n", text)
    text = _EXCESS_NEWLINES.sub("\n\n", text)
    return text.strip()
