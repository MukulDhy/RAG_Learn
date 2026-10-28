
import hashlib
import json
from pathlib import Path

import pymupdf


# Project paths
PDF_DIR = Path(r"ai_ml_knowledge_base")
OUTPUT_DIR = Path(r"rag_project\src\rag_project\ingest\data")
OUTPUT_FILE = OUTPUT_DIR / "pages.jsonl"


def calculate_sha256(file_path: Path) -> str:
    """Create a stable fingerprint for a PDF file."""
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)

    return sha256.hexdigest()


def extract_pdf(pdf_path: Path) -> list[dict]:
    """Extract text from a PDF while preserving page metadata."""
    records = []
    file_hash = calculate_sha256(pdf_path)

    with pymupdf.open(pdf_path) as document:
        if document.needs_pass:
            raise ValueError("PDF requires a password")

        metadata = document.metadata or {}

        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text", sort=True)
            text = text.replace("\x00", "").strip()

            records.append(
                {
                    "document_id": file_hash,
                    "document_name": pdf_path.name,
                    "page_number": page_number,
                    "title": metadata.get("title") or pdf_path.stem,
                    "text": text,
                    "character_count": len(text),
                    "needs_ocr": len(text) < 50,
                }
            )

    return records


def main() -> None:
    if not PDF_DIR.exists():
        raise FileNotFoundError(
            f"PDF directory not found: {PDF_DIR.resolve()}"
        )

    pdf_files = sorted(
        path for path in PDF_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found inside {PDF_DIR.resolve()}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_pages = 0
    empty_pages = 0
    failed_files = []

    # Write to a temporary file so an interrupted run does not
    # replace a previous successful output with partial results.
    temp_file = OUTPUT_FILE.with_suffix(".jsonl.tmp")

    try:
        with temp_file.open("w", encoding="utf-8") as output:
            for pdf_path in pdf_files:
                try:
                    records = extract_pdf(pdf_path)

                    for record in records:
                        output.write(
                            json.dumps(record, ensure_ascii=False)
                            + "\n"
                        )

                        total_pages += 1
                        if record["needs_ocr"]:
                            empty_pages += 1

                    print(
                        f"OK: {pdf_path.name} "
                        f"({len(records)} pages)"
                    )

                except Exception as error:
                    failed_files.append(pdf_path.name)
                    print(f"FAILED: {pdf_path.name}: {error}")

        if failed_files:
            raise RuntimeError(
                f"{len(failed_files)} PDF(s) failed. "
                "Fix them and rerun ingestion. "
                f"Failed files: {failed_files}"
            )

        temp_file.replace(OUTPUT_FILE)

    finally:
        if temp_file.exists():
            temp_file.unlink()

    print("\n--- Ingestion summary ---")
    print(f"PDFs processed: {len(pdf_files)}")
    print(f"Pages extracted: {total_pages}")
    print(f"Pages flagged for OCR review: {empty_pages}")
    print(f"Output: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
