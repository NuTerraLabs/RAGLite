# src/contexttape/ingest_generic.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Generator, Iterable, Optional, Tuple

# Optional PDF dependency
try:
    from pypdf import PdfReader  # type: ignore
    _HAS_PYPDF = True
except Exception:
    _HAS_PYPDF = False


_TEXT_EXTS = {
    ".txt", ".md", ".rst", ".log", ".cfg", ".conf", ".ini", ".csv", ".tsv",
    ".json", ".yaml", ".yml", ".toml", ".py", ".js", ".ts", ".java", ".go",
    ".rb", ".php", ".c", ".cc", ".cpp", ".h", ".hpp", ".sh", ".bash", ".zsh",
    ".html", ".htm", ".css"
}
_PDF_EXT = ".pdf"


def _read_text_file(path: Path, max_bytes: Optional[int]) -> str:
    size = path.stat().st_size
    to_read = min(size, max_bytes) if max_bytes is not None else size
    with path.open("rb") as f:
        data = f.read(to_read)
    # best effort decode
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(enc, errors="ignore")
        except Exception:
            continue
    return data.decode("utf-8", errors="ignore")


def _read_pdf_file(path: Path, max_pages: Optional[int]) -> str:
    if not _HAS_PYPDF:
        return ""
    out = []
    reader = PdfReader(str(path))
    pages = reader.pages
    end = len(pages) if max_pages is None else min(len(pages), max_pages)
    for i in range(end):
        try:
            out.append(pages[i].extract_text() or "")
        except Exception:
            # tolerate damaged pages
            continue
    return "\n".join(out).strip()


def iter_files(
    root: str | os.PathLike,
    exts: Optional[Iterable[str]] = None,
    max_bytes: Optional[int] = None,
    max_pdf_pages: Optional[int] = 20,
    follow_symlinks: bool = False,
) -> Generator[Tuple[Path, str], None, None]:
    """
    Walk a directory tree and yield (path, text) for supported files.

    Parameters
    ----------
    root : directory to walk
    exts : whitelist of extensions (case-insensitive). If None, defaults to _TEXT_EXTS + .pdf
    max_bytes : read cap for text-like files
    max_pdf_pages : limit number of pages per PDF (if PyPDF installed)
    follow_symlinks : whether to follow symlinks

    Yields
    ------
    (Path, str)
    """
    allow = set(e.lower() for e in (exts or list(_TEXT_EXTS | {_PDF_EXT})))
    root_path = Path(root)

    if root_path.is_file():
        # Single file mode
        ext = root_path.suffix.lower()
        if ext in allow:
            if ext == _PDF_EXT:
                text = _read_pdf_file(root_path, max_pdf_pages)
            else:
                text = _read_text_file(root_path, max_bytes)
            if text.strip():
                yield (root_path, text)
        return

    for dirpath, dirnames, filenames in os.walk(root_path, followlinks=follow_symlinks):
        # You can prune dirs here if needed
        for fname in filenames:
            p = Path(dirpath) / fname
            ext = p.suffix.lower()
            if ext not in allow:
                continue
            try:
                if ext == _PDF_EXT:
                    text = _read_pdf_file(p, max_pdf_pages)
                else:
                    text = _read_text_file(p, max_bytes)
            except Exception:
                continue
            if text and text.strip():
                yield (p, text)
