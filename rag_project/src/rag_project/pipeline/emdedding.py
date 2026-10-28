
import hashlib
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


INPUT_FILE = Path(r"rag_project\src\rag_project\pipeline\ingest\data\processed\chunks.jsonl")
OUTPUT_DIR = Path(r"rag_project\src\rag_project\pipeline\ingest\data\processed")
EMBEDDINGS_FILE = OUTPUT_DIR / "embeddings.npy"
INDEX_FILE = OUTPUT_DIR / "embedded_chunks.jsonl"
MANIFEST_FILE = OUTPUT_DIR / "embedding_manifest.json"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 16
DEVICE = "cpu"


def load_chunks(path: Path) -> list[dict]:
    chunks = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error

            text = chunk.get("text", "").strip()

            if not text:
                continue

            chunks.append(chunk)

    return chunks


def validate_model_token_limits(
    chunks: list[dict],
    model: SentenceTransformer,
) -> None:
    """
    Check actual model-tokenizer lengths before encoding.
    Avoid silently embedding truncated chunks.
    """
    tokenizer = model.tokenizer
    max_length = model.max_seq_length
    problems = []

    for index, chunk in enumerate(chunks):
        token_ids = tokenizer(
            chunk["text"],
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )["input_ids"]

        if len(token_ids) > max_length:
            problems.append({
                "index": index,
                "chunk_id": chunk.get("chunk_id"),
                "document_name": chunk.get("document_name"),
                "page_number": chunk.get("page_number"),
                "actual_tokens": len(token_ids),
            })

            if len(problems) >= 10:
                break

    if problems:
        raise ValueError(
            f"Found chunks exceeding the model's "
            f"{max_length}-token input limit. "
            f"First problematic chunks: {problems}. "
            "Reduce CHUNK_SIZE in chunk.py (for example, "
            "to 350-400), rerun chunk.py, and retry."
        )


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_FILE}. Run chunk.py first."
        )

    chunks = load_chunks(INPUT_FILE)

    if not chunks:
        raise ValueError("No non-empty chunks to embed.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {MODEL_NAME}")

    model = SentenceTransformer(
        MODEL_NAME,
        device=DEVICE,
    )

    print(f"Device: {DEVICE}")
    print(f"Model max sequence length: {model.max_seq_length}")
    print(f"Chunks to embed: {len(chunks)}")

    validate_model_token_limits(chunks, model)

    texts = [chunk["text"] for chunk in chunks]

    print("Generating embeddings...")

    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    vectors = np.asarray(vectors, dtype=np.float32)

    if vectors.ndim != 2:
        raise ValueError(
            f"Expected a 2D embedding matrix, got {vectors.shape}"
        )

    if vectors.shape[0] != len(chunks):
        raise ValueError(
            "Embedding count does not match chunk count."
        )

    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain NaN or infinite values.")

    norms = np.linalg.norm(vectors, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError("Expected normalized embedding vectors.")

    # Use the same ordering for both files. Row i in the NumPy
    # matrix belongs to record i in embedded_chunks.jsonl.
    embedded_records = []

    for index, chunk in enumerate(chunks):
        embedded_records.append({
            **chunk,
            "embedding_index": index,
        })

    # Write temporary files first so an interrupted run is less
    # likely to leave a partial output looking like a complete run.
    temp_vectors = OUTPUT_DIR / "embeddings.tmp.npy"
    temp_index = OUTPUT_DIR / "embedded_chunks.tmp.jsonl"
    temp_manifest = OUTPUT_DIR / "embedding_manifest.tmp.json"

    try:
        with temp_vectors.open("wb") as file:
            np.save(file, vectors, allow_pickle=False)

        with temp_index.open("w", encoding="utf-8") as file:
            for record in embedded_records:
                file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

        manifest = {
            "model_name": MODEL_NAME,
            "device": DEVICE,
            "dimensions": int(vectors.shape[1]),
            "dtype": str(vectors.dtype),
            "normalized": True,
            "similarity_metric": "cosine",
            "batch_size": BATCH_SIZE,
            "chunk_count": len(chunks),
            "embedding_matrix_shape": list(vectors.shape),
            "source_file": str(INPUT_FILE),
            "source_sha256": hashlib.sha256(
                INPUT_FILE.read_bytes()
            ).hexdigest(),
        }

        temp_manifest.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        temp_vectors.replace(EMBEDDINGS_FILE)
        temp_index.replace(INDEX_FILE)
        temp_manifest.replace(MANIFEST_FILE)

    finally:
        for path in (temp_vectors, temp_index, temp_manifest):
            if path.exists():
                path.unlink()

    print("\n--- Embedding Summary ---")
    print(f"Chunks embedded: {vectors.shape[0]}")
    print(f"Dimensions: {vectors.shape[1]}")
    print(f"Vector data type: {vectors.dtype}")
    print(f"Vector matrix shape: {vectors.shape}")
    print(f"Embeddings: {EMBEDDINGS_FILE}")
    print(f"Metadata: {INDEX_FILE}")
    print(f"Manifest: {MANIFEST_FILE}")


if __name__ == "__main__":
    main()
