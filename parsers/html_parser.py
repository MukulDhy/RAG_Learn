from pathlib import Path

from bs4 import BeautifulSoup

from schemas import DocumentRecord, ParseResult


def _readable_text(element) -> str:
    lines = []
    for line in element.get_text("\n", strip=True).splitlines():
        normalized_line = " ".join(line.split())
        if normalized_line and (not lines or lines[-1] != normalized_line):
            lines.append(normalized_line)
    return "\n".join(lines)


def parse_html(file_path: str) -> ParseResult:
    path = Path(file_path)

    if not path.is_file():
        return ParseResult(error=f"File not found: {file_path}")

    try:
        html = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")

        title_element = soup.find("title")
        title = title_element.get_text(" ", strip=True) if title_element else None

        description_element = soup.find(
            "meta",
            attrs={"name": lambda value: value and value.lower() == "description"},
        )
        description = (
            description_element.get("content", "").strip()
            if description_element
            else None
        )

        for element in soup.select(
            "script, style, noscript, template, nav, footer, aside"
        ):
            element.decompose()

        content_element = soup.find("main")
        selector = "main"
        if content_element is None:
            content_element = soup.find("article")
            selector = "article"
        if content_element is None:
            content_element = soup.body or soup
            selector = "body" if soup.body else "document"

        metadata = {
            "file_type": "html",
            "content_selector": selector,
        }
        if title:
            metadata["title"] = title
        if description:
            metadata["description"] = description

        document = DocumentRecord(
            content=_readable_text(content_element),
            source=file_path,
            metadata=metadata,
        )
        return ParseResult(success=True, documents=[document])

    except Exception as exc:
        return ParseResult(error=f"Failed to parse '{file_path}': {exc}")
