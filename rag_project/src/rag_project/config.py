"""Central configuration.

Everything tunable lives here so experiments change one file, not scattered
constants. Values can be overridden with environment variables prefixed with
``RAG_`` (for example ``RAG_CHUNK_SIZE=300``) or a local ``.env`` file.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: The project root (the directory containing ``pyproject.toml``).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Extensions we know how to load. Everything else is skipped during ingestion.
SUPPORTED_EXTENSIONS = (".pdf", ".html", ".htm", ".txt")


class Settings(BaseSettings):
    """Runtime settings, loaded from environment variables and ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- Locations ---------------------------------------------------------
    #: The document corpus. Defaults to the sibling ``ai_ml_knowledge_base``.
    kb_dir: Path = PROJECT_ROOT.parent / "ai_ml_knowledge_base"
    #: Root for processed data, indexes and reports (safe to delete/rebuild).
    data_dir: Path = PROJECT_ROOT / "data"

    # -- Chunking ----------------------------------------------------------
    #: Target chunk size in tokens. bge-small truncates at 512 tokens, so
    #: staying well below that keeps whole chunks searchable.
    chunk_size: int = 400
    #: Tokens repeated between neighbouring chunks so facts on a boundary
    #: are not lost.
    chunk_overlap: int = 80
    #: Pages with less extracted text than this are reported as suspicious
    #: (usually scanned/image-only PDFs that would need OCR).
    min_page_chars: int = 50

    # -- Embeddings --------------------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 32

    # -- Retrieval ---------------------------------------------------------
    #: Results returned to the caller.
    top_k: int = 8
    #: Candidates fetched before MMR re-ranking; must be >= top_k.
    candidate_k: int = 40
    #: MMR trade-off: 1.0 = pure relevance, 0.0 = pure diversity.
    mmr_lambda: float = 0.7

    @model_validator(mode="after")
    def _check_sizes(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be >= top_k")
        return self

    # -- Derived paths -----------------------------------------------------
    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def chunks_path(self) -> Path:
        """All chunks produced by ingestion, before embedding."""
        return self.processed_dir / "chunks.jsonl"

    @property
    def vectors_path(self) -> Path:
        """L2-normalized embedding matrix, one row per chunk."""
        return self.index_dir / "vectors.npy"

    @property
    def index_chunks_path(self) -> Path:
        """Chunk metadata aligned row-by-row with ``vectors_path``."""
        return self.index_dir / "chunks.jsonl"

    @property
    def manifest_path(self) -> Path:
        """Content hashes used to skip unchanged files on re-ingest."""
        return self.index_dir / "manifest.json"

    def ensure_dirs(self) -> None:
        """Create the writable data directories."""
        for path in (self.processed_dir, self.index_dir, self.reports_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings instance (one read of the environment)."""
    return Settings()
