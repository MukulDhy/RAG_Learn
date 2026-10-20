import hashlib
import re

def _fingerprint(text: str) -> str:
    canonical = " ".join(text.split()).casefold()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def deduplicate_paragraphs(
    text: str, *, mode: str = "adjacent", min_chars: int = 40,
) -> tuple[str, list[dict[str, object]]]:
    """Remove duplicate paragraphs conservatively; global mode is opt-in."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if mode not in {"adjacent", "global"}:
        raise ValueError("mode must be 'adjacent' or 'global'")
    if min_chars < 0:
        raise ValueError("min_chars must be >= 0")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]
    kept, removed, seen = [], [], set()
    for index, paragraph in enumerate(paragraphs):
        fp = _fingerprint(paragraph)
        eligible = len(paragraph) >= min_chars
        duplicate = (
            eligible and bool(kept) and _fingerprint(kept[-1]) == fp
            if mode == "adjacent" else eligible and fp in seen
        )
        if duplicate:
            removed.append({"paragraph_index": index, "text": paragraph,
                            "fingerprint": fp, "mode": mode})
        else:
            kept.append(paragraph)
            seen.add(fp)
    return "\n\n".join(kept), removed
