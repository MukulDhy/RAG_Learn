import re

_BULLET = re.compile(r"^\s*(?:[-*+•▪◦]|\d+[.)]|[A-Za-z][.)])\s+")
_TABLE = re.compile(r"\|.*\|")
_HEADING = re.compile(r"^(?:#{1,6}\s|[A-Z][A-Z0-9 ,:&()/\-]{2,})$")

def _structural(line: str) -> bool:
    s = line.strip()
    return bool(
        _BULLET.match(s) or _TABLE.search(s) or
        s.startswith(("```", "~~~")) or _HEADING.match(s)
    )

def join_wrapped_lines(text: str) -> str:
    """Join likely wrapped plain-text lines; preserve blank paragraphs and structures."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out, paragraph = [], []
    fenced = False

    def flush():
        if paragraph:
            out.append(" ".join(x.strip() for x in paragraph if x.strip()))
            paragraph.clear()

    for raw in lines:
        line = raw.strip()
        if line.startswith(("```", "~~~")):
            flush()
            out.append(line)
            fenced = not fenced
            continue
        if fenced:
            flush()
            out.append(raw.rstrip())
            continue
        if not line:
            flush()
            if out and out[-1] != "":
                out.append("")
            continue
        if _structural(line):
            flush()
            out.append(line)
            continue
        if paragraph and paragraph[-1].endswith("-"):
            paragraph[-1] = paragraph[-1][:-1] + line
        else:
            paragraph.append(line)

    flush()
    while out and out[0] == "":
        out.pop(0)
    while out and out[-1] == "":
        out.pop()
    compact = []
    for item in out:
        if item != "" or (compact and compact[-1] != ""):
            compact.append(item)
    return "\n".join(compact)
