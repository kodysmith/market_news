"""Book/text ingestion — reads documents and feeds them through the concept extractor."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks, breaking at paragraph boundaries."""
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) > chunk_size and current:
            chunks.append(current.strip())
            # Keep overlap from end of current chunk
            words = current.split()
            overlap_words = words[-overlap // 5:] if len(words) > overlap // 5 else words
            current = " ".join(overlap_words) + "\n\n" + para
        else:
            current = current + "\n\n" + para if current else para

    if current.strip():
        chunks.append(current.strip())

    return chunks


def read_file(path: str) -> str:
    """Read a text file. Supports .txt, .md, .pdf (with pymupdf), .epub."""
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix in (".txt", ".md"):
        return p.read_text(encoding="utf-8")

    if suffix == ".pdf":
        return _read_pdf(p)

    if suffix == ".epub":
        return _read_epub(p)

    # Fallback — try as text
    return p.read_text(encoding="utf-8")


def _read_pdf(path: Path) -> str:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("Install PyMuPDF to read PDFs: pip install pymupdf")

    doc = fitz.open(str(path))
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n\n".join(pages)


def _read_epub(path: Path) -> str:
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("Install ebooklib and beautifulsoup4 to read EPUBs: pip install ebooklib beautifulsoup4")

    book = epub.read_epub(str(path))
    chapters = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = soup.get_text()
        if text.strip():
            chapters.append(text.strip())
    return "\n\n".join(chapters)


def discover_books(directory: str, extensions: Optional[List[str]] = None) -> List[Path]:
    """Find all readable book files in a directory."""
    if extensions is None:
        extensions = [".txt", ".md", ".pdf", ".epub"]
    root = Path(directory)
    files = []
    for ext in extensions:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)
