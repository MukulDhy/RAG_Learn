from technova_rag_preprocessing.paragraph_joiner import join_wrapped_lines

def test_joins_wrapped_lines():
    assert join_wrapped_lines("This is a sentence that was\nwrapped by PDF.") == "This is a sentence that was wrapped by PDF."

def test_preserves_paragraphs():
    assert join_wrapped_lines("First paragraph.\n\nSecond paragraph.") == "First paragraph.\n\nSecond paragraph."

def test_hyphenated_wrap():
    assert join_wrapped_lines("infor-\nmation") == "information"

def test_preserves_lists():
    assert join_wrapped_lines("- First item\n- Second item") == "- First item\n- Second item"

def test_preserves_tables():
    text = "| A | B |\n|---|---|\n| 1 | 2 |"
    assert join_wrapped_lines(text) == text

def test_preserves_code():
    text = "```python\nx = 1\ny = 2\n```"
    assert join_wrapped_lines(text) == text
