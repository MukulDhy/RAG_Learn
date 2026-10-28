
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


DATA_DIR = Path(r"rag_project\src\rag_project\pipeline\ingest\data\processed")

EMBEDDINGS_FILE = DATA_DIR / "embeddings.npy"
CHUNKS_FILE = DATA_DIR / "embedded_chunks.jsonl"
MANIFEST_FILE = DATA_DIR / "embedding_manifest.json"

TOP_K = 5


def load_chunks(path: Path) -> list[dict]:
    chunks = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}"
                ) from error

    return chunks


def load_index():
    for path in (
        EMBEDDINGS_FILE,
        CHUNKS_FILE,
        MANIFEST_FILE,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run embed.py first."
            )

    manifest = json.loads(
        MANIFEST_FILE.read_text(encoding="utf-8")
    )

    chunks = load_chunks(CHUNKS_FILE)

    # Disable pickle for safe loading of numerical data.
    vectors = np.load(
        EMBEDDINGS_FILE,
        allow_pickle=False,
    )

    if vectors.ndim != 2:
        raise ValueError("Expected a 2D vector matrix.")

    if vectors.shape[0] != len(chunks):
        raise ValueError(
            "Vector count and metadata count do not match."
        )

    if vectors.shape[1] != manifest["dimensions"]:
        raise ValueError(
            "Vector dimensions disagree with the manifest."
        )

    expected_indices = list(range(len(chunks)))
    actual_indices = [
        chunk["embedding_index"] for chunk in chunks
    ]

    if actual_indices != expected_indices:
        raise ValueError(
            "Chunk metadata ordering is inconsistent."
        )

    return vectors, chunks, manifest


def retrieve(
    question: str,
    model: SentenceTransformer,
    vectors: np.ndarray,
    chunks: list[dict],
    top_k: int = TOP_K,
) -> list[dict]:
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    # Embed the query using the same model used for documents.
    query_vector = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    query_vector = np.asarray(
        query_vector,
        dtype=np.float32,
    )

    if query_vector.shape[0] != vectors.shape[1]:
        raise ValueError(
            "Query and document embedding dimensions differ."
        )

    # All document vectors and the query vector are normalized.
    # Their dot product therefore equals cosine similarity.
    scores = vectors @ query_vector

    # Get the best candidates without sorting every score.
    k = min(top_k, len(chunks))
    candidate_indices = np.argpartition(
        scores, len(scores) - k
    )[-k:]

    # Sort those candidates by descending similarity.
    ranked_indices = candidate_indices[
        np.argsort(scores[candidate_indices])[::-1]
    ]

    results = []

    for rank, index in enumerate(ranked_indices, start=1):
        chunk = chunks[int(index)]

        results.append({
            "rank": rank,
            "score": float(scores[index]),
            "document_name": chunk["document_name"],
            "page_number": chunk["page_number"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
        })

    return results


def main() -> None:
    vectors, chunks, manifest = load_index()

    model_name = manifest["model_name"]

    print(f"Loading embedding model: {model_name}")

    model = SentenceTransformer(
        model_name,
        device="cpu",
    )

    if model.get_sentence_embedding_dimension() != vectors.shape[1]:
        raise ValueError(
            "Loaded model output dimensions do not match "
            "the stored vectors."
        )

    print(f"Indexed chunks: {len(chunks)}")
    print(f"Vector dimensions: {vectors.shape[1]}")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Question: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            continue

        results = retrieve(
            question=question,
            model=model,
            vectors=vectors,
            chunks=chunks,
            top_k=TOP_K,
        )

        print("\n--- Retrieved passages ---")

        for result in results:
            print(
                f"\nRank: {result['rank']} | "
                f"Similarity: {result['score']:.4f}"
            )
            print(
                f"Source: {result['document_name']} | "
                f"Page: {result['page_number']}"
            )
            print(f"Chunk ID: {result['chunk_id']}")
            print("-" * 60)
            print(result["text"])
            print("-" * 60)

        print()


if __name__ == "__main__":
    main()
