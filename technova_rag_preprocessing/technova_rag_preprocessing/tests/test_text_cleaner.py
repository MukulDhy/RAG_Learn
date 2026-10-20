import pytest
from technova_rag_preprocessing.text_cleaner import normalize_text

@pytest.mark.parametrize(("source", "expected"), [
    ("   Hello   world   ", "Hello world"),
    ("First paragraph.\n\n\n\nSecond paragraph.", "First paragraph.\n\nSecond paragraph."),
    ("Line one\nLine two", "Line one\nLine two"),
    ("   \n \t  ", ""),
    ("A\t\tB\n\nC", "A B\n\nC"),
    ("A\r\nB", "A\nB"), ("A\rB", "A\nB"),
    ("A  \nB", "A\nB"), ("", ""), ("  café\u00a0text  ", "café text"),
])
def test_normalize_text(source, expected):
    assert normalize_text(source) == expected

def test_idempotent():
    result = normalize_text("  A\t B\n\n\nC  ")
    assert normalize_text(result) == result

def test_rejects_non_string():
    with pytest.raises(TypeError):
        normalize_text(None)
