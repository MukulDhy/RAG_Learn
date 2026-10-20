from technova_rag_preprocessing.header_footer import remove_repeated_headers_footers
from technova_rag_preprocessing.models import PageText

def test_removes_repeated_header_and_page_number():
    pages = [PageText(1, f"ACME POLICY\nPage {i}\n\nContent {i}.") for i in range(1, 4)]
    cleaned, audit = remove_repeated_headers_footers(pages)
    assert all("ACME POLICY" not in p.text for p in cleaned)
    assert all("Content" in p.text for p in cleaned)
    assert len(audit) >= 3

def test_one_page_unchanged():
    pages = [PageText(1, "Header\nContent\nFooter")]
    cleaned, audit = remove_repeated_headers_footers(pages)
    assert cleaned == pages
    assert audit == []

def test_middle_repeated_clause_kept():
    pages = [PageText(1, "Header A\nLine two\nLine three\nImportant repeated clause\nLine five\nLine six\nFooter A"),
             PageText(2, "Header B\nLine two\nLine three\nImportant repeated clause\nLine five\nLine six\nFooter B")]
    cleaned, _ = remove_repeated_headers_footers(pages)
    assert all("Important repeated clause" in p.text for p in cleaned)
