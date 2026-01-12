# src/contexttape/relevance.py
from __future__ import annotations
from typing import List, Tuple, Optional
from .utils import norm_tokens

def lexical_overlap_score(query: str, text: str) -> float:
    """
    Deterministic [0,1] score using Jaccard overlap of normalized tokens.
    Downweights trivial 1–2 token queries a bit to avoid false positives.
    """
    qt = set(norm_tokens(query))
    tt = set(norm_tokens(text))
    if not qt or not tt:
        return 0.0
    inter = len(qt & tt)
    union = len(qt | tt)
    base = inter / union if union else 0.0
    if len(qt) <= 2:
        base *= 0.9
    return base

def hybrid_score(vec_score: float, lex_score: float, alpha: float = 0.7) -> float:
    """
    α·vector + (1−α)·lexical, both clamped to [0,1].
    """
    v = max(0.0, min(1.0, vec_score))
    l = max(0.0, min(1.0, lex_score))
    return alpha * v + (1.0 - alpha) * l

def select_relevant_blocks(
    query: str,
    hits: List[Tuple[str, float, int, int]],
    wiki_store,
    chat_store,
    *,
    topk: int,
    min_vec: float = 0.32,
    min_lex: float = 0.12,
    min_hybrid: float = 0.28,
    alpha: float = 0.7,
    max_blocks: Optional[int] = None,
    max_preview_chars: int = 1200,
) -> Tuple[bool, List[str]]:
    """
    Given raw hits [(src, vec_score, tid, eid), ...], build context blocks ONLY
    for entries that pass relevance gates. Returns (use_context, blocks).
    """
    blocks: List[str] = []
    keep = 0
    limit = max_blocks or topk

    for i, (src, vec_score, tid, eid) in enumerate(hits, 1):
        if keep >= limit:
            break
        store = wiki_store if src == "wiki" else chat_store
        text = store.read_text(tid)
        lex = lexical_overlap_score(query, text)
        hyb = hybrid_score(vec_score, lex, alpha=alpha)

        passes = (vec_score >= min_vec) or (lex >= min_lex) or (hyb >= min_hybrid)
        if not passes:
            continue

        base = wiki_store.dir_path if src == "wiki" else chat_store.dir_path
        preview = text[:max_preview_chars]
        blocks.append(
            f"[{src.upper()} {len(blocks)+1}] (vec={vec_score:.4f}, lex={lex:.4f}, hyb={hyb:.4f}, tid={tid}, eid={eid})\n"
            f"(text_path={base}/segment_{tid}.is | vec_path={base}/segment_{eid}.is)\n\n{preview}"
        )
        keep += 1

    return (len(blocks) > 0, blocks)
