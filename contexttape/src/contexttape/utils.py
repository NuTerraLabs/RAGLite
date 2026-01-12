# src/contexttape/utils.py
import os
import re

_WORD_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def norm_tokens(s: str) -> list[str]:
    """
    Lowercase, strip punctuation, split into alnum tokens.
    Keeps behavior deterministic and cheap.
    """
    s = s.lower()
    return _WORD_RE.findall(s)
