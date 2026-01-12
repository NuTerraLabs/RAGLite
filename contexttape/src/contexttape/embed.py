# embed.py
from __future__ import annotations
import os
from typing import Iterable, List, Literal, Tuple, Optional

import numpy as np
import tiktoken
from openai import OpenAI


# -----------------------------
# Client
# -----------------------------
def get_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY")
    return OpenAI(api_key=key)


# -----------------------------
# Tokenization helpers
# -----------------------------
_EMBED_MODEL = "text-embedding-3-small"
_EMBED_DIM = 1536
# As of OpenAI docs, this model supports up to ~8192 tokens. Keep a buffer.
_MODEL_MAX_TOKENS = int(os.getenv("EMBED_MODEL_MAX_TOKENS", "8192"))

def _enc():
    # Use the model-specific encoding if available
    try:
        return tiktoken.encoding_for_model(_EMBED_MODEL)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")

def _encode(text: str) -> List[int]:
    return _enc().encode(text)

def _decode(tokens: List[int]) -> str:
    return _enc().decode(tokens)


def _chunk_tokens(
    tokens: List[int],
    window: int,
    overlap: int,
) -> List[List[int]]:
    if window <= 0:
        raise ValueError("window must be > 0")
    if not (0 <= overlap < window):
        raise ValueError("overlap must satisfy 0 <= overlap < window")

    chunks: List[List[int]] = []
    i = 0
    step = window - overlap
    while i < len(tokens):
        chunks.append(tokens[i : i + window])
        i += step
    return chunks


# -----------------------------
# Pooling / aggregation
# -----------------------------
_Pool = Literal["mean", "max", "weighted"]

def _pool_vectors(vectors: List[np.ndarray], weights: Optional[List[float]], mode: _Pool) -> np.ndarray:
    if not vectors:
        # Empty input: return zeros
        return np.zeros((_EMBED_DIM,), dtype=np.float32)

    mat = np.vstack(vectors)  # [n, d]

    if mode == "mean":
        return mat.mean(axis=0).astype(np.float32)

    if mode == "max":
        return mat.max(axis=0).astype(np.float32)

    if mode == "weighted":
        if not weights or len(weights) != len(vectors):
            # Fallback to mean if weights are missing/mismatched
            return mat.mean(axis=0).astype(np.float32)
        w = np.array(weights, dtype=np.float32).reshape(-1, 1)  # [n, 1]
        w = w / (w.sum() + 1e-8)
        return (mat * w).sum(axis=0).astype(np.float32)

    raise ValueError(f"Unknown pool mode: {mode}")


# -----------------------------
# Core: dynamic, chunked embedding
# -----------------------------
def embed_text_1536(
    client: OpenAI,
    text: str,
    *,
    # If text <= model max, we embed in one shot. Otherwise we chunk by tokens:
    chunk_tokens: int = int(os.getenv("EMBED_CHUNK_TOKENS", "800")),  # per-chunk size
    chunk_overlap: int = int(os.getenv("EMBED_CHUNK_OVERLAP", "200")),  # overlap between chunks
    pool: _Pool = "mean",  # "mean" | "max" | "weighted"
    return_per_chunk: bool = False,  # if True, return list of chunk vectors instead of pooled
    verbose: bool = False,
) -> np.ndarray | List[np.ndarray]:
    """
    Dynamically embed arbitrarily long text.
    - If the text token length <= model limit: one-shot embed.
    - Else: windowed token chunks with overlap, per-chunk embedding, then pool.

    Env overrides:
      EMBED_MODEL_MAX_TOKENS (default 8192)
      EMBED_CHUNK_TOKENS (default 800)
      EMBED_CHUNK_OVERLAP (default 200)
    """
    tokens = _encode(text)
    n_tokens = len(tokens)

    if n_tokens == 0:
        if verbose:
            print(f"[EMBED] {_EMBED_MODEL} len=0 (empty input)")
        return np.zeros((_EMBED_DIM,), dtype=np.float32) if not return_per_chunk else []

    # Case 1: fits in one call
    if n_tokens <= _MODEL_MAX_TOKENS:
        if verbose:
            print(f"[EMBED] model={_EMBED_MODEL} len={n_tokens} (one-shot)")
        payload = text
        r = client.embeddings.create(model=_EMBED_MODEL, input=payload)
        vec = np.asarray(r.data[0].embedding, dtype=np.float32)
        return vec if not return_per_chunk else [vec]

    # Case 2: chunk & pool
    if verbose:
        print(f"[EMBED] model={_EMBED_MODEL} total_len={n_tokens} -> chunking window={chunk_tokens} overlap={chunk_overlap}")

    token_chunks = _chunk_tokens(tokens, window=chunk_tokens, overlap=chunk_overlap)
    texts = [_decode(ch) for ch in token_chunks]

    # Batch-friendly call (OpenAI embeddings supports batching list input)
    r = client.embeddings.create(model=_EMBED_MODEL, input=texts)
    vecs: List[np.ndarray] = [np.asarray(item.embedding, dtype=np.float32) for item in r.data]

    if return_per_chunk:
        if verbose:
            print(f"[EMBED] chunks={len(vecs)} (returned per-chunk)")
        return vecs

    # default weights = chunk lengths (weighted pooling)
    weights = [len(ch) for ch in token_chunks] if pool == "weighted" else None
    pooled = _pool_vectors(vecs, weights=weights, mode=pool)
    if verbose:
        print(f"[EMBED] chunks={len(vecs)} pooled={pool}")
    return pooled


# -----------------------------
# Image & audio embeddings (placeholder deterministic)
# -----------------------------
def embed_image(image_bytes: bytes) -> np.ndarray:
    """
    Replace with a real vision-embedding model if you have one.
    Deterministic placeholder: hash -> RNG -> 1536-d vector.
    """
    import hashlib
    h = hashlib.sha256(image_bytes).digest()
    seed = int.from_bytes(h[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, _EMBED_DIM).astype(np.float32)

def embed_audio(audio_bytes: bytes, sample_rate_hint: int | None = None) -> np.ndarray:
    """
    Replace with a real audio-embedding model if you have one.
    Deterministic placeholder: hash -> RNG -> 1536-d vector.
    """
    import hashlib
    h = hashlib.sha256(audio_bytes).digest()
    seed = int.from_bytes(h[8:16], "little", signed=False)
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, _EMBED_DIM).astype(np.float32)
