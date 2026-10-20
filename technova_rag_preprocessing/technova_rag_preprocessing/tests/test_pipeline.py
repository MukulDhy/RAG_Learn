from technova_rag_preprocessing.models import PageText
from technova_rag_preprocessing.pipeline import clean_pages

def test_pipeline():
    pages = [
        PageText(1, "ACME POLICY\nPage 1\n\nThis is the first wrapped\nline of content."),
        PageText(2, "ACME POLICY\nPage 2\n\nThis is the second page of content."),
    ]
    result = clean_pages(pages)
    assert all("ACME POLICY" not in p.text for p in result.pages)
    assert "first wrapped line of content." in result.pages[0].text
    assert result.removed_headers_footers
