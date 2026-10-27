# rag_project

A from-scratch retrieval-augmented generation (RAG) system over the
`ai_ml_knowledge_base` folder. Every stage is written plainly so the mechanics
stay visible: no chain-of-abstraction framework.

## Status

| Phase | What | Status |
|-------|------|--------|
| 1 | Scaffold: layout, config, tests | done |
| 2 | Ingestion: parse, clean, chunk | next |
| 3 | Embeddings + NumPy vector index | pending |
| 4 | Retrieval: top-k, filters, MMR | pending |
| 5 | Evaluation: recall@k, MRR | pending |
| 6 | Streamlit search UI | pending |
| 7 | Tuning: chunk size, hybrid search | pending |
| 8 | Documentation + walkthrough | pending |

Generation is deliberately deferred; the system answers with *retrieved
context* first, then an LLM can be layered on top later.

## Setup

```powershell
cd D:\RAG\rag_project
uv sync
uv run pytest
```

## Configuration

Settings live in `src/rag_project/config.py`. Override any of them with a
`RAG_`-prefixed environment variable or a `.env` file (see `.env.example`).

| Setting | Default | Meaning |
|---------|---------|---------|
| `RAG_KB_DIR` | `../ai_ml_knowledge_base` | Document corpus |
| `RAG_DATA_DIR` | `./data` | Generated chunks, indexes, reports |
| `RAG_CHUNK_SIZE` | `400` | Target tokens per chunk |
| `RAG_CHUNK_OVERLAP` | `80` | Tokens shared by neighbours |
| `RAG_TOP_K` | `8` | Results returned |
| `RAG_CANDIDATE_K` | `40` | Candidates before MMR |
| `RAG_MMR_LAMBDA` | `0.7` | Relevance vs. diversity |

## Layout

```
src/rag_project/
  config.py          settings and paths
  ingest/            loading, cleaning, chunking (phase 2)
  index/             embeddings and vector index (phase 3)
  retrieve/          search and MMR (phase 4)
  evaluation/        retrieval metrics (phase 5)
tests/               pytest suite
scripts/             runnable entry points (phase 2+)
app/                 Streamlit UI (phase 6)
data/                generated, gitignored
```

## Why these defaults

- **400-token chunks** keep text under bge-small's 512-token limit, so no
  chunk is silently truncated before embedding.
- **80-token overlap** protects facts that straddle a chunk boundary.
- **NumPy brute-force search** is plenty for a few thousand chunks and makes
  every part of ranking inspectable.
