from typing import Any

from pydantic import BaseModel, Field


class DocumentRecord(BaseModel):
    content: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParseResult(BaseModel):
    success: bool = False
    documents: list[DocumentRecord] = Field(default_factory=list)
    error: str | None = None
