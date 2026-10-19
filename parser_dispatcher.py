from collections.abc import Callable
from pathlib import Path

from parsers.html_parser import parse_html
from parsers.pdf_parser import parse_pdf
from schemas import DocumentRecord, ParseResult


def _parse_pdf(file_path: str) -> ParseResult:
    pdf_result = parse_pdf(file_path)
    empty_page_numbers = {page.page_number for page in pdf_result.empty_pages}

    documents = [
        DocumentRecord(
            content=page.text,
            source=file_path,
            metadata={
                "page_number": page.page_number,
                "file_type": "pdf",
                "empty": page.page_number in empty_page_numbers,
            },
        )
        for page in pdf_result.pages
    ]

    return ParseResult(
        success=pdf_result.success,
        documents=documents,
        error=pdf_result.error,
    )


_PARSERS: dict[str, Callable[[str], ParseResult]] = {
    ".pdf": _parse_pdf,
    ".html": parse_html,
    ".htm": parse_html,
}


def parse_file(file_path: str) -> ParseResult:
    path = Path(file_path)

    if not path.is_file():
        return ParseResult(error=f"File not found: {file_path}")

    parser = _PARSERS.get(path.suffix.lower())
    if parser is None:
        return ParseResult(error=f"Unsupported file type: {path.suffix or '<none>'}")

    try:
        return parser(file_path)
    except Exception as exc:
        return ParseResult(error=f"Failed to parse '{file_path}': {exc}")
