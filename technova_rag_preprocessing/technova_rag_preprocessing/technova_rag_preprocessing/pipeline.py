from .deduplication import deduplicate_paragraphs
from .header_footer import remove_repeated_headers_footers
from .models import CleaningResult, PageText
from .paragraph_joiner import join_wrapped_lines
from .text_cleaner import normalize_text

def clean_pages(pages: list[PageText], *, join_paragraph_lines: bool = True,
                deduplicate_mode: str = "adjacent") -> CleaningResult:
    """Normalize pages, remove repeated edge artifacts, then join/deduplicate."""
    normalized = [PageText(p.page_number, normalize_text(p.text), p.line_y_positions)
                  for p in pages]
    no_artifacts, artifact_audit = remove_repeated_headers_footers(normalized)
    output, duplicate_audit = [], []
    for page in no_artifacts:
        text = join_wrapped_lines(page.text) if join_paragraph_lines else page.text
        text, removed = deduplicate_paragraphs(text, mode=deduplicate_mode)
        for item in removed:
            item["page_number"] = page.page_number
        duplicate_audit.extend(removed)
        output.append(PageText(page.page_number, text, page.line_y_positions))
    return CleaningResult(output, artifact_audit, duplicate_audit)
