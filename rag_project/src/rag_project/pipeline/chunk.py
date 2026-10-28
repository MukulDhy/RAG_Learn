
import hashlib
import json
import re
import statistics
from pathlib import Path

import tiktoken


INPUT_FILE = Path(r"rag_project\src\rag_project\pipeline\ingest\data\pages_cleaned.jsonl")
OUTPUT_FILE = Path(r"rag_project\src\rag_project\pipeline\ingest\data/processed/chunks.jsonl")

CHUNK_SIZE = 340
CHUNK_OVERLAP = 75

# A practical baseline tokenizer. Validate against your final
# embedding model before deploying.
TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))


def load_pages(path: Path) -> list[dict]:
    pages = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                pages.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {error}"
                ) from error

    return pages


def split_recursive(
    text: str,
    max_tokens: int,
    separators: tuple[str, ...] = (
        "\n\n", "\n", ". ", "; ", ", ", " "
    ),
) -> list[str]:
    """
    Split text recursively, preferring meaningful boundaries.
    Fall back to smaller boundaries when a section is too large.
    """
    text = text.strip()

    if not text:
        return []

    if count_tokens(text) <= max_tokens:
        return [text]

    if not separators:
        # Last resort: split by tokens when a single unbroken
        # string or code block exceeds the budget.
        token_ids = TOKENIZER.encode(text)
        pieces = []

        for start in range(0, len(token_ids), max_tokens):
            piece = TOKENIZER.decode(
                token_ids[start:start + max_tokens]
            ).strip()

            if piece:
                pieces.append(piece)

        return pieces

    separator = separators[0]

    if separator not in text:
        return split_recursive(
            text, max_tokens, separators[1:]
        )

    # Keep the delimiter attached to the preceding piece so
    # punctuation and paragraph boundaries aren't discarded.
    raw_parts = re.split(
        f"(?<={re.escape(separator)})",
        text,
    )

    parts = [part for part in raw_parts if part.strip()]
    result = []
    current = ""

    for part in parts:
        candidate = current + part

        if count_tokens(candidate) <= max_tokens:
            current = candidate
            continue

        if current.strip():
            result.append(current.strip())

        if count_tokens(part) <= max_tokens:
            current = part
        else:
            result.extend(
                split_recursive(
                    part, max_tokens, separators[1:]
                )
            )
            current = ""

    if current.strip():
        result.append(current.strip())

    # Safety check: every result must respect the token budget.
    final_result = []

    for piece in result:
        if count_tokens(piece) <= max_tokens:
            final_result.append(piece)
        else:
            final_result.extend(
                split_recursive(piece, max_tokens, separators[1:])
            )

    return final_result


def add_overlap(
    pieces: list[str],
    overlap_tokens: int,
    max_tokens: int,
) -> list[str]:
    """
    Add a suffix of the previous chunk to the next chunk.

    The overlap is trimmed when necessary so the combined chunk
    never exceeds max_tokens.
    """
    if overlap_tokens <= 0 or len(pieces) < 2:
        return pieces

    result = [pieces[0]]

    for piece in pieces[1:]:
        previous_tokens = TOKENIZER.encode(result[-1])
        current_tokens = TOKENIZER.encode(piece)

        # Use only the trailing tokens from the preceding chunk.
        overlap = TOKENIZER.decode(
            previous_tokens[-overlap_tokens:]
        ).strip()

        # Avoid overlap consuming the whole current chunk.
        while overlap and count_tokens(overlap + "\n" + piece) > max_tokens:
            overlap_ids = TOKENIZER.encode(overlap)

            if len(overlap_ids) <= 1:
                overlap = ""
                break

            overlap = TOKENIZER.decode(overlap_ids[1:]).strip()

        combined = (
            f"{overlap}\n{piece}".strip()
            if overlap
            else piece
        )

        # Extremely unusual tokenization edge case: fall back
        # to the original piece if the combination still exceeds
        # the configured maximum.
        if count_tokens(combined) > max_tokens:
            combined = piece

        result.append(combined)

    return result


def make_chunk_id(
    document_id: str,
    page_number: int,
    chunk_index: int,
    text: str,
) -> str:
    identity = (
        f"{document_id}:{page_number}:{chunk_index}:"
        f"{text}"
    )

    return hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()


def chunk_page(page: dict) -> list[dict]:
    text = page.get("text", "").strip()

    if not text:
        return []

    document_id = page.get("document_id", "unknown")
    page_number = page.get("page_number", 0)
    document_name = page.get("document_name", "unknown.pdf")

    base_pieces = split_recursive(text, CHUNK_SIZE)
    pieces = add_overlap(
        base_pieces,
        CHUNK_OVERLAP,
        CHUNK_SIZE,
    )

    chunks = []

    for index, piece in enumerate(pieces):
        token_count = count_tokens(piece)

        chunk = {
            "chunk_id": make_chunk_id(
                document_id, page_number, index, piece
            ),
            "document_id": document_id,
            "document_name": document_name,
            "page_number": page_number,
            "chunk_index": index,
            "text": piece,
            "token_count": token_count,
            "content_hash": hashlib.sha256(
                piece.encode("utf-8")
            ).hexdigest(),
        }

        chunks.append(chunk)

    return chunks


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_FILE}. Run clean.py first."
        )

    pages = load_pages(INPUT_FILE)

    if not pages:
        raise ValueError("No pages found in cleaned input.")

    all_chunks = []

    for page in pages:
        page_chunks = chunk_page(page)
        all_chunks.extend(page_chunks)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for chunk in all_chunks:
            file.write(
                json.dumps(chunk, ensure_ascii=False) + "\n"
            )

    token_counts = [
        chunk["token_count"] for chunk in all_chunks
    ]

    print("\n--- Chunking Summary ---")
    print(f"Pages processed: {len(pages)}")
    print(f"Chunks created: {len(all_chunks)}")

    if token_counts:
        print(f"Minimum tokens: {min(token_counts)}")
        print(f"Maximum tokens: {max(token_counts)}")
        print(
            f"Average tokens: "
            f"{statistics.mean(token_counts):.1f}"
        )
        print(
            f"Median tokens: "
            f"{statistics.median(token_counts):.1f}"
        )

    print(f"Chunk output: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
