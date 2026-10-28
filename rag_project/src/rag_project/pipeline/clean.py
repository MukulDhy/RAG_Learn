
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


INPUT_FILE = Path(r"rag_project\src\rag_project\pipeline\ingest\data\pages.jsonl")
OUTPUT_FILE = Path(r"rag_project\src\rag_project\pipeline\ingest\data\pages_cleaned.jsonl")
REPORT_FILE = Path(r"rag_project\src\rag_project\pipeline\ingest\data\quality_report.json")


def load_jsonl(path: Path) -> list[dict]:
    """Read JSONL and report malformed records with line numbers."""
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error

    return records


def normalize_line(line: str) -> str:
    """Normalize a line for comparison, not for displayed content."""
    line = re.sub(r"\s+", " ", line).strip()
    return line.casefold()


def clean_text(text: str) -> str:
    """
    Apply conservative text cleanup.
    Preserve punctuation, numbers, equations, and code.
    """
    text = text.replace("\x00", "")
    text = text.replace("\u00a0", " ")  # Non-breaking space
    text = text.replace("\u00ad", "")   # Soft hyphen character

    # Normalize line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    cleaned_lines = []

    for line in text.split("\n"):
        # Collapse repeated horizontal whitespace, not line breaks.
        line = re.sub(r"[^\S\n]+", " ", line).strip()
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Limit excessive blank lines without flattening paragraphs.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def content_hash(text: str) -> str:
    """Fingerprint normalized content to detect exact duplicate pages."""
    normalized = normalize_line(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def analyze_page(text: str) -> dict:
    """Calculate useful extraction-quality indicators."""
    total_chars = len(text)
    words = re.findall(r"\b\w+\b", text, flags=re.UNICODE)

    replacement_chars = text.count("\ufffd")
    control_chars = sum(
        1 for char in text
        if ord(char) < 32 and char not in "\n\t"
    )

    suspicious_chars = sum(
        1 for char in text
        if char in "\ufffd\u25a1\u25af"
    )

    return {
        "character_count": total_chars,
        "word_count": len(words),
        "replacement_char_count": replacement_chars,
        "control_char_count": control_chars,
        "suspicious_char_count": suspicious_chars,
        "low_text": total_chars < 50,
        "high_symbol_ratio": (
            suspicious_chars / total_chars > 0.05
            if total_chars else False
        ),
    }


def find_repeated_headers_footers(
    records: list[dict],
) -> dict[str, set[str]]:
    """
    Identify lines repeatedly appearing near page boundaries.

    Candidates must appear in at least 3 pages and at least
    60% of a document's pages. This is a heuristic: review
    candidates before using aggressive removal in production.
    """
    by_document = defaultdict(list)

    for record in records:
        by_document[record["document_id"]].append(record)

    result = {}

    for document_id, pages in by_document.items():
        # Very short documents are not suitable for this heuristic.
        if len(pages) < 3:
            result[document_id] = set()
            continue

        boundary_counts = Counter()

        for page in pages:
            lines = [
                normalize_line(line)
                for line in page.get("text", "").splitlines()
                if line.strip()
            ]

            # Examine the first and last three non-empty lines.
            boundary_lines = lines[:3] + lines[-3:]

            # Count a line at most once per page.
            for line in set(boundary_lines):
                if len(line) >= 5:
                    boundary_counts[line] += 1

        minimum_count = max(3, int(len(pages) * 0.60 + 0.999))

        result[document_id] = {
            line
            for line, count in boundary_counts.items()
            if count >= minimum_count
        }

    return result


def remove_repeated_boundary_lines(
    text: str,
    repeated_lines: set[str],
) -> tuple[str, list[str]]:
    """Remove only recognized repeated boundary lines."""
    lines = text.splitlines()
    non_empty_indices = [
        index for index, line in enumerate(lines) if line.strip()
    ]

    boundary_indices = set(
        non_empty_indices[:3] + non_empty_indices[-3:]
    )

    removed = []
    output = []

    for index, line in enumerate(lines):
        normalized = normalize_line(line)

        if (
            index in boundary_indices
            and normalized in repeated_lines
        ):
            removed.append(line)
            continue

        output.append(line)

    cleaned = "\n".join(output)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    return cleaned, removed


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_FILE}. Run ingest.py first."
        )

    records = load_jsonl(INPUT_FILE)

    if not records:
        raise ValueError("The input file contains no page records.")

    # First pass: identify recurring header/footer candidates.
    repeated_lines_by_document = find_repeated_headers_footers(
        records
    )

    cleaned_records = []
    duplicate_groups = defaultdict(list)
    issues = []
    total_removed_lines = 0

    # Second pass: clean and inspect each page.
    for record in records:
        original_text = record.get("text", "")
        document_id = record.get("document_id", "unknown")
        page_number = record.get("page_number", 0)

        text = clean_text(original_text)

        text, removed_lines = remove_repeated_boundary_lines(
            text,
            repeated_lines_by_document.get(document_id, set()),
        )

        total_removed_lines += len(removed_lines)

        metrics = analyze_page(text)
        page_hash = content_hash(text)

        # Keep duplicate pages; report them instead of deleting them.
        duplicate_groups[page_hash].append(
            {
                "document_name": record.get("document_name"),
                "page_number": page_number,
            }
        )

        output_record = {
            **record,
            "text": text,
            "character_count": len(text),
            "content_hash": page_hash,
            "quality": metrics,
            "removed_boundary_lines": removed_lines,
        }

        cleaned_records.append(output_record)

        if metrics["low_text"]:
            issues.append({
                "document_name": record.get("document_name"),
                "page_number": page_number,
                "issue": "low_text",
                "character_count": metrics["character_count"],
            })

        if (
            metrics["replacement_char_count"] > 0
            or metrics["control_char_count"] > 0
            or metrics["high_symbol_ratio"]
        ):
            issues.append({
                "document_name": record.get("document_name"),
                "page_number": page_number,
                "issue": "suspicious_extraction",
                "replacement_char_count":
                    metrics["replacement_char_count"],
                "control_char_count": metrics["control_char_count"],
                "high_symbol_ratio": metrics["high_symbol_ratio"],
            })

    duplicates = [
        {
            "content_hash": page_hash,
            "pages": pages,
        }
        for page_hash, pages in duplicate_groups.items()
        if len(pages) > 1
    ]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for record in cleaned_records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    report = {
        "input_file": str(INPUT_FILE),
        "output_file": str(OUTPUT_FILE),
        "total_pages": len(records),
        "low_text_pages": sum(
            record["quality"]["low_text"]
            for record in cleaned_records
        ),
        "suspicious_pages": sum(
            any(
                issue["document_name"] == record.get("document_name")
                and issue["page_number"] == record.get("page_number")
                and issue["issue"] == "suspicious_extraction"
                for issue in issues
            )
            for record in cleaned_records
        ),
        "pages_in_duplicate_groups": sum(
            len(group["pages"]) for group in duplicates
        ),
        "duplicate_groups": duplicates,
        "repeated_boundary_lines_removed": total_removed_lines,
        "issues": issues,
    }

    REPORT_FILE.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n--- Quality Analysis Summary ---")
    print(f"Pages analyzed: {len(records)}")
    print(f"Low-text pages: {report['low_text_pages']}")
    print(f"Suspicious pages: {report['suspicious_pages']}")
    print(f"Duplicate groups: {len(duplicates)}")
    print(
        "Pages in duplicate groups: "
        f"{report['pages_in_duplicate_groups']}"
    )
    print(
        "Repeated boundary lines removed: "
        f"{total_removed_lines}"
    )
    print(f"Cleaned data: {OUTPUT_FILE}")
    print(f"Quality report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
