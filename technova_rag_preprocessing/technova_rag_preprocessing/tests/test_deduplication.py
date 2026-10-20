import pytest
from technova_rag_preprocessing.deduplication import deduplicate_paragraphs

LONG = "This is a sufficiently long paragraph used to test duplicate extraction behavior."

def test_adjacent_duplicate_removed():
    result, removed = deduplicate_paragraphs(f"{LONG}\n\n{LONG}")
    assert result == LONG
    assert len(removed) == 1

def test_non_adjacent_repeat_preserved_by_default():
    text = f"{LONG}\n\nDifferent paragraph.\n\n{LONG}"
    result, removed = deduplicate_paragraphs(text)
    assert result.count(LONG) == 2
    assert removed == []

def test_global_mode_is_opt_in():
    text = f"{LONG}\n\nDifferent paragraph.\n\n{LONG}"
    result, removed = deduplicate_paragraphs(text, mode="global")
    assert result.count(LONG) == 1
    assert len(removed) == 1

def test_short_repeats_kept():
    result, removed = deduplicate_paragraphs("Yes.\n\nYes.")
    assert result == "Yes.\n\nYes."
    assert removed == []

def test_invalid_mode():
    with pytest.raises(ValueError):
        deduplicate_paragraphs("text", mode="magic")
