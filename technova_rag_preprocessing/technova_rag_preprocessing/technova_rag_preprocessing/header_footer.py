    from collections import defaultdict
import re
from .models import PageText

_PAGE_NUMBER = re.compile(r"^(?:page\s+)?\d+(?:\s*(?:of|/)\s*\d+)?$", re.I)

def _canonical(line: str) -> str:
    line = " ".join(line.split()).strip()
    if _PAGE_NUMBER.match(line):
        return "<PAGE_NUMBER>"
    return line.casefold()

def remove_repeated_headers_footers(
    pages: list[PageText], *, edge_lines: int = 3,
    min_page_fraction: float = 0.6, min_repetitions: int = 2,
) -> tuple[list[PageText], list[dict[str, object]]]:
    """Remove repeated page-edge lines using frequency and edge-position evidence."""
    if not isinstance(pages, list):
        raise TypeError("pages must be a list")
    if edge_lines < 1:
        raise ValueError("edge_lines must be >= 1")
    if not 0 < min_page_fraction <= 1:
        raise ValueError("min_page_fraction must be in (0, 1]")
    if min_repetitions < 2:
        raise ValueError("min_repetitions must be >= 2")
    if len(pages) < 2:
        return pages.copy(), []

    occurrences = defaultdict(set)
    page_lines = []
    page_edges = []
    for pi, page in enumerate(pages):
        lines = page.text.splitlines()
        page_lines.append(lines)
        nonempty = [i for i, line in enumerate(lines) if line.strip()]
        edges = set(nonempty[:edge_lines] + nonempty[-edge_lines:])
        page_edges.append(edges)
        for i in edges:
            key = _canonical(lines[i])
            if key:
                occurrences[key].add(pi)

    min_pages = max(min_repetitions, int(len(pages) * min_page_fraction + 0.999999))
    candidates = {k for k, ids in occurrences.items() if len(ids) >= min_pages}
    cleaned, audit = [], []
    for pi, page in enumerate(pages):
        kept = []
        for i, line in enumerate(page_lines[pi]):
            key = _canonical(line)
            if i in page_edges[pi] and key in candidates:
                audit.append({"page_number": page.page_number, "line": line,
                              "canonical": key, "line_index": i})
            else:
                kept.append(line)
        cleaned.append(PageText(page.page_number, "\n".join(kept).strip(),
                                page.line_y_positions))
    return cleaned, audit
