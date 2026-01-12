from __future__ import annotations

import re
from typing import List, Tuple, Optional
import numpy as np
from .storage import ISStore, MultiStore

_token_re = re.compile(r"[A-Za-z0-9_]+")

def _tokens(s: str) -> List[str]:
    return [t.lower() for t in _token_re.findall(s)]

def lexical_overlap(query: str, text: str) -> float:
    qt = set(_tokens(query))
    if not qt:
        return 0.0
    tt = set(_tokens(text))
    return len(qt & tt) / max(1, len(qt))

def hybrid_score(alpha: float, cos: float, lex: float) -> float:
    return alpha * cos + (1.0 - alpha) * lex

def role_bias(text: str, bias_assistant: float = 0.02, bias_user: float = -0.01) -> float:
    # small nudges per doc suggestion: prefer assistant snippets slightly
    if text.startswith("assistant:"):
        return bias_assistant
    if text.startswith("user:"):
        return bias_user
    return 0.0

def combined_search(
    query: str,
    qvec: np.ndarray,
    wiki_store: ISStore,
    chat_store: ISStore,
    top_k: int = 5,
    alpha: float = 0.7,
    stride: int = 1,
    min_vec: float = 0.0,
    min_lex: float = 0.0,
    min_hybrid: float = 0.0,
    use_shards: bool = False,
) -> List[Tuple[str, float, int, int]]:
    """
    Returns list of (src, final_score, tid, eid)
    """
    results: List[Tuple[str, float, int, int]] = []

    if use_shards:
        ms = MultiStore([wiki_store, chat_store])
        per = ms.search(qvec, per_shard_k=max(8, top_k), final_k=top_k, stride=stride, coarse_limit=16)
        for dir_path, cos, tid, eid in per:
            src = "wiki" if dir_path == wiki_store.dir_path else "chat"
            text = (wiki_store if src == "wiki" else chat_store).read_text(tid)
            lex = lexical_overlap(query, text)
            final = hybrid_score(alpha, cos, lex) + role_bias(text)
            if cos >= min_vec and lex >= min_lex and final >= min_hybrid:
                results.append((src, final, tid, eid))
    else:
        for src, store in (("wiki", wiki_store), ("chat", chat_store)):
            hits = store.search_by_vector(qvec, top_k=max(8, top_k), stride=stride, coarse_limit=16)
            for (cos, tid, eid) in hits:
                text = store.read_text(tid)
                lex = lexical_overlap(query, text)
                final = hybrid_score(alpha, cos, lex) + role_bias(text)
                if cos >= min_vec and lex >= min_lex and final >= min_hybrid:
                    results.append((src, final, tid, eid))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

def select_relevant_blocks(
    user: str,
    hits: List[Tuple[str, float, int, int]],
    wiki_store: ISStore,
    chat_store: ISStore,
    max_blocks: int = 5,
    max_preview_chars: int = 1200,
) -> Tuple[bool, List[str]]:
    """
    Assemble context blocks for the top results; return (use_context, blocks)
    """
    blocks: List[str] = []
    for i, (src, score, tid, eid) in enumerate(hits[:max_blocks], 1):
        store = wiki_store if src == "wiki" else chat_store
        base = store.dir_path
        text = store.read_text(tid)[:max_preview_chars]
        block = (
            f"[{src.upper()} {i}] (score={score:.4f}, tid={tid}, eid={eid})\n"
            f"(text_path={base}/segment_{tid}.is | vec_path={base}/segment_{eid}.is)\n\n{text}"
        )
        blocks.append(block)
    return (len(blocks) > 0, blocks)
