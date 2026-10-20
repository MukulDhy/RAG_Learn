# TechNova RAG preprocessing (RAG-104 to RAG-107)

Python 3.10+. Runtime uses only the standard library; tests use pytest.

Run from this directory:
```bash
python -m pip install pytest
python -m pytest -q
```

## Design and approaches

- RAG-104 `normalize_text`: fixed number of linear regex passes; O(n) time and O(n) output memory. Plain text only; don't use blindly on code/fixed-width tables.
- RAG-105 `join_wrapped_lines`: joins lines within paragraphs; blank lines, bullets, headings, Markdown table rows, and fenced code are treated as boundaries. Heuristics require corpus-specific tests.
- RAG-106 `remove_repeated_headers_footers`: a line must repeat across enough pages and occur in the top/bottom edge band. Page numbers are canonicalized. Audit records show removals. Layout coordinates and document-specific configuration improve accuracy.
- RAG-107 `deduplicate_paragraphs`: defaults to adjacent duplicates only. Global exact deduplication is opt-in because repeated legal clauses and instructions can be legitimate.
- Pipeline keeps page boundaries until header/footer detection is done and returns audit lists.

## Integration

```python
from technova_rag_preprocessing import PageText, clean_pages

result = clean_pages([
    PageText(1, "ACME POLICY\\nPage 1\\n\\nWrapped text\\ncontinues here."),
    PageText(2, "ACME POLICY\\nPage 2\\n\\nSecond page text."),
])
for page in result.pages:
    print(page.page_number, page.text)
print(result.removed_headers_footers)
```

Retain the original parser output for auditing. This is a conservative baseline, not a universal layout reconstruction engine.
