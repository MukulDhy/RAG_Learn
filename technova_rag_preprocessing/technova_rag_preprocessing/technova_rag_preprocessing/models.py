from dataclasses import dataclass, field

@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str
    line_y_positions: tuple[float, ...] = ()

@dataclass
class CleaningResult:
    pages: list[PageText]
    removed_headers_footers: list[dict[str, object]] = field(default_factory=list)
    removed_duplicates: list[dict[str, object]] = field(default_factory=list)
