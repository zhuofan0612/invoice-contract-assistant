"""Load a contract from disk into (metadata, body text).

Markdown is the native format for this dataset. The PDF path is implemented
because real procurement contracts arrive as PDFs, including scanned ones: text
is taken from the embedded text layer where one exists, tables are recovered
with pdfplumber, and pages with no text layer fall back to OCR.

Honest scoping: the shipped dataset is Markdown, so the PDF/OCR branch is
exercised by its interface and by any PDF you drop in, not by the golden set.
C-2024-006 carries `scanned: true` to mark it as the OCR provenance case.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


@dataclass
class ParsedDocument:
    path: Path
    metadata: dict
    body: str
    source_mode: str = "native"  # native | pdf-text | pdf-ocr
    warnings: list[str] = field(default_factory=list)


def parse_contract(path: Path) -> ParsedDocument:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown", ".txt"}:
        return _parse_markdown(path)
    if suffix == ".pdf":
        return _parse_pdf(path)
    raise ValueError(f"Unsupported contract format: {path.name}")


def _parse_markdown(path: Path) -> ParsedDocument:
    raw = path.read_text(encoding="utf-8")
    metadata, body = _split_frontmatter(raw)
    mode = "native"
    warnings: list[str] = []
    if metadata.get("scanned"):
        # Provenance only: the shipped fixture is text. A real scanned PDF with
        # the same frontmatter would be routed through the OCR branch below.
        mode = "pdf-ocr"
        warnings.append(
            "Contract is marked scanned; treated as OCR provenance "
            "(fixture is text, so no OCR was actually run)."
        )
    return ParsedDocument(path=path, metadata=metadata, body=body, source_mode=mode, warnings=warnings)


def _split_frontmatter(raw: str) -> tuple[dict, str]:
    if not raw.startswith("---"):
        return {}, raw
    parts = raw.split("---", 2)
    if len(parts) < 3:
        return {}, raw
    metadata = yaml.safe_load(parts[1]) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    return metadata, parts[2].lstrip("\n")


def _parse_pdf(path: Path) -> ParsedDocument:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "PDF ingestion needs PyMuPDF. Install the parsing extra: "
            "pip install -r requirements-parsing.txt"
        ) from exc

    warnings: list[str] = []
    pages: list[str] = []
    ocr_pages = 0

    with fitz.open(path) as doc:
        for page_no, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if not text:
                text = _ocr_page(path, page_no, warnings)
                if text:
                    ocr_pages += 1
            tables = _extract_tables(path, page_no, warnings)
            pages.append("\n\n".join(filter(None, [text, *tables])))

    mode = "pdf-ocr" if ocr_pages else "pdf-text"
    return ParsedDocument(
        path=path,
        metadata={"source_pdf": path.name},
        body="\n\n".join(pages),
        source_mode=mode,
        warnings=warnings,
    )


def _ocr_page(path: Path, page_no: int, warnings: list[str]) -> str:
    """OCR a page that has no text layer."""
    try:
        import fitz
        import pytesseract
        from PIL import Image
    except ImportError:
        warnings.append(
            f"page {page_no} has no text layer and OCR deps are missing "
            "(pytesseract, pillow); page skipped"
        )
        return ""

    try:
        with fitz.open(path) as doc:
            pix = doc[page_no - 1].get_pixmap(dpi=300)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        # swe+eng: these are Swedish municipal contracts.
        return pytesseract.image_to_string(image, lang="swe+eng").strip()
    except Exception as exc:  # pragma: no cover - environment dependent
        warnings.append(f"OCR failed on page {page_no}: {exc}")
        return ""


def _extract_tables(path: Path, page_no: int, warnings: list[str]) -> list[str]:
    """Recover tables as Markdown so rate tables survive chunking intact."""
    try:
        import pdfplumber
    except ImportError:
        return []

    rendered: list[str] = []
    try:
        with pdfplumber.open(path) as pdf:
            for table in pdf.pages[page_no - 1].extract_tables() or []:
                rows = [[(cell or "").strip() for cell in row] for row in table if row]
                if len(rows) < 2:
                    continue
                header, *body = rows
                rendered.append(
                    "\n".join(
                        [
                            "| " + " | ".join(header) + " |",
                            "|" + "|".join(["---"] * len(header)) + "|",
                            *["| " + " | ".join(r) + " |" for r in body],
                        ]
                    )
                )
    except Exception as exc:  # pragma: no cover - environment dependent
        warnings.append(f"table extraction failed on page {page_no}: {exc}")
    return rendered
