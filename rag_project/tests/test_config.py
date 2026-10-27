from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_project.config import PROJECT_ROOT, SUPPORTED_EXTENSIONS, Settings


def test_defaults_are_sane() -> None:
    settings = Settings()

    assert settings.chunk_size == 400
    assert settings.chunk_overlap == 80
    assert settings.chunk_overlap < settings.chunk_size
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.kb_dir == PROJECT_ROOT.parent / "ai_ml_knowledge_base"
    assert ".pdf" in SUPPORTED_EXTENSIONS


def test_derived_paths_live_under_data_dir(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)

    assert settings.processed_dir == tmp_path / "processed"
    assert settings.chunks_path == tmp_path / "processed" / "chunks.jsonl"
    assert settings.vectors_path == tmp_path / "index" / "vectors.npy"
    assert settings.manifest_path == tmp_path / "index" / "manifest.json"


def test_ensure_dirs_creates_every_directory(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)

    settings.ensure_dirs()

    for path in (settings.processed_dir, settings.index_dir, settings.reports_dir):
        assert path.is_dir()


def test_overlap_larger_than_chunk_size_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(chunk_size=100, chunk_overlap=200)


def test_candidate_k_must_cover_top_k() -> None:
    with pytest.raises(ValidationError):
        Settings(top_k=10, candidate_k=5)


def test_environment_variables_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_CHUNK_SIZE", "250")
    monkeypatch.setenv("RAG_TOP_K", "3")

    settings = Settings()

    assert settings.chunk_size == 250
    assert settings.top_k == 3
