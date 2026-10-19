import pymupdf
from pydantic import BaseModel, Field


class Page(BaseModel):
    page_number: int
    text: str = ""


class Result(BaseModel):
    success: bool = False
    pages: list[Page] = Field(default_factory=list)
    empty_pages: list[Page] = Field(default_factory=list)
    error: str | None = None


def parse_pdf(file_path: str) -> Result:
    result = Result()
    try:
        with pymupdf.open(file_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                single_page = Page(page_number=page_num, text=text)

                result.pages.append(single_page)
                if not text:
                    result.empty_pages.append(single_page)

        result.success = True

    except Exception as exc:
        result.error = str(exc)

    return result
