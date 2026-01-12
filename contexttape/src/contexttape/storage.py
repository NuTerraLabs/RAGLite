from __future__ import annotations

"""
ContextTape Storage Module
==========================

This module provides the core storage abstraction for ContextTape - a zero-infrastructure
RAG (Retrieval-Augmented Generation) system that stores text and embeddings in simple
binary segment files.

Key Classes:
    - ISHeader: 32-byte binary header for each segment
    - ISStore: Single-directory segment store
    - MultiStore: Multi-directory store with late fusion search

Data Types:
    - DT_TEXT (0): UTF-8 text segments
    - DT_VEC_F32 (1): Float32 vector embeddings  
    - DT_VEC_I8 (2): Int8 quantized vectors (4x smaller)
    - DT_COARSE (100): Coarse index segments for prefiltering
    - DT_IMAGE (10): Image blob storage
    - DT_AUDIO (11): Audio blob storage
    - DT_BLOB (12): Generic binary blob
    - DT_JSON (13): JSON-serialized metadata

Example Usage:
    >>> from contexttape import ISStore
    >>> import numpy as np
    >>> 
    >>> # Create or open a store
    >>> store = ISStore("my_store")
    >>> 
    >>> # Add text with embedding
    >>> embedding = np.random.randn(1536).astype(np.float32)
    >>> text_id, vec_id = store.append_text_with_embedding(
    ...     "Hello world", embedding, quantize=True
    ... )
    >>> 
    >>> # Search by vector
    >>> query = np.random.randn(1536).astype(np.float32)
    >>> for score, tid, eid in store.search_by_vector(query, top_k=5):
    ...     print(f"{score:.4f}: {store.read_text(tid)[:50]}...")

File Format:
    Each .is segment file contains:
    - 32-byte header (see ISHeader)
    - Variable-length payload (text, vector, or blob)

    Header layout (little-endian):
    | Field     | Type    | Bytes | Description                    |
    |-----------|---------|-------|--------------------------------|
    | next_id   | int32   | 4     | Link to paired segment         |
    | prev_id   | int32   | 4     | Back-link to paired segment    |
    | data_len  | int32   | 4     | Payload byte length            |
    | data_type | int32   | 4     | Segment type (DT_* constants)  |
    | dim       | int32   | 4     | Vector dimension (0 for text)  |
    | scale     | float32 | 4     | Quantization scale factor      |
    | reserved  | bytes   | 8     | Timestamp (int64)              |
"""

import os
import glob
import struct
import time
import mmap
import io
from typing import List, Tuple, Dict, Optional, Iterable, Union
import numpy as np

# ========== Core header / dt types ==========
HEADER_SIZE = 32
# next_i32, prev_i32, data_len_i32, data_type_i32, dim_i32, scale_f32, reserved_8B
HEADER_FMT = "<iiiiif8s"

DT_TEXT     = 0
DT_VEC_F32  = 1
DT_VEC_I8   = 2
DT_COARSE   = 100   # "special index segments" (e.g., IVF centroids / coarse codes)
DT_IMAGE   = 10
DT_AUDIO   = 11
DT_BLOB    = 12   # arbitrary bytes (e.g., original media)
DT_JSON    = 13   # structured text payload (JSON-serialized UTF-8)

# small helper to write arbitrary bytes as a segment
def _ensure_bytes(obj) -> bytes:
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj)
    if isinstance(obj, str):
        return obj.encode("utf-8")
    raise TypeError("Expected bytes or str")

# ========== Optional security hooks (no-op by default) ==========
def _encrypt(b: bytes) -> bytes:   # replace with real AEAD if desired
    return b

def _decrypt(b: bytes) -> bytes:   # replace with real AEAD if desired
    return b

class ISHeader:
    __slots__ = ("next_id", "prev_id", "data_len", "data_type", "dim", "scale", "ts")
    def __init__(self, next_id=-1, prev_id=-1, data_len=0, data_type=DT_TEXT, dim=0, scale=1.0, ts: Optional[int]=None):
        self.next_id = next_id
        self.prev_id = prev_id
        self.data_len = data_len
        self.data_type = data_type
        self.dim = dim
        self.scale = scale
        self.ts = int(time.time()) if ts is None else int(ts)

    def pack(self) -> bytes:
        # encode ts (int64) into reserved 8 bytes
        reserved = struct.pack("<q", self.ts)
        return struct.pack(HEADER_FMT, self.next_id, self.prev_id, self.data_len,
                           self.data_type, self.dim, self.scale, reserved)

    @staticmethod
    def unpack(b: bytes) -> "ISHeader":
        ni, pi, dl, dt, dm, sc, res = struct.unpack(HEADER_FMT, b)
        (ts,) = struct.unpack("<q", res)
        return ISHeader(ni, pi, dl, dt, dm, sc, ts)

class ISStore:
    """
    Single-directory store. Segments: segment_<id>.is
    DT_TEXT: UTF-8 text
    DT_VEC_F32: float32 vector
    DT_VEC_I8: int8 vector with scale
    DT_COARSE: opaque coarse index payload (centroids/codes)
    """
    def __init__(self, dir_path: str):
        self.dir_path = dir_path
        os.makedirs(self.dir_path, exist_ok=True)
        self._next_id = self._discover_next_id()

    # ---------- paths ----------
    def _segment_path(self, seg_id: int) -> str:
        return os.path.join(self.dir_path, f"segment_{seg_id}.is")

    def _glob_segments(self) -> List[int]:
        ids = []
        for p in glob.glob(os.path.join(self.dir_path, "segment_*.is")):
            base = os.path.basename(p)
            try:
                sid = int(base.split("_")[1].split(".")[0])
                ids.append(sid)
            except Exception:
                continue
        return sorted(ids)

    def _discover_next_id(self) -> int:
        ids = self._glob_segments()
        return (max(ids) + 1) if ids else 0

    @property
    def next_id(self) -> int:
        return self._next_id

    # ---------- low level I/O ----------
    def _write_segment(self, seg_id: int, header: ISHeader, payload: bytes) -> None:
        with open(self._segment_path(seg_id), "wb") as f:
            f.write(header.pack())
            f.write(_encrypt(payload))

    def _read_header(self, seg_id: int) -> ISHeader:
        with open(self._segment_path(seg_id), "rb") as f:
            return ISHeader.unpack(f.read(HEADER_SIZE))

    def _read_payload(self, seg_id: int, data_len: int) -> bytes:
        path = self._segment_path(seg_id)
        with open(path, "rb") as f:
            mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
            try:
                # header at [0:32], payload immediately after
                start = HEADER_SIZE
                end = HEADER_SIZE + data_len
                return _decrypt(mm[start:end])
            finally:
                mm.close()

    # ---------- append ----------
    def append_text(self, text: str) -> int:
        seg_id = self._next_id
        payload = text.encode("utf-8")
        hdr = ISHeader(-1, -1, len(payload), DT_TEXT, 0, 1.0)
        self._write_segment(seg_id, hdr, payload)
        self._next_id += 1
        return seg_id

    def append_vector_f32(self, vec: np.ndarray, prev_text_id: Optional[int] = None, scale: float = 1.0) -> int:
        if vec.dtype != np.float32:
            vec = vec.astype(np.float32, copy=False)
        seg_id = self._next_id
        payload = vec.tobytes(order="C")
        hdr = ISHeader(-1, prev_text_id if prev_text_id is not None else -1,
                       len(payload), DT_VEC_F32, int(vec.shape[0]), float(scale))
        self._write_segment(seg_id, hdr, payload)
        self._next_id += 1
        return seg_id

    def append_vector_i8(self, vec_f32: np.ndarray, prev_text_id: Optional[int] = None) -> int:
        vec = vec_f32.astype(np.float32, copy=False)
        scale = float(np.max(np.abs(vec)) / 127.0) if np.max(np.abs(vec)) > 0 else 1.0
        q = np.clip(np.round(vec / scale), -127, 127).astype(np.int8, copy=False)
        seg_id = self._next_id
        payload = q.tobytes(order="C")
        hdr = ISHeader(-1, prev_text_id if prev_text_id is not None else -1,
                       len(payload), DT_VEC_I8, int(vec.shape[0]), scale)
        self._write_segment(seg_id, hdr, payload)
        self._next_id += 1
        return seg_id

    def append_coarse_segment(self, payload: bytes) -> int:
        seg_id = self._next_id
        hdr = ISHeader(-1, -1, len(payload), DT_COARSE, 0, 1.0)
        self._write_segment(seg_id, hdr, payload)
        self._next_id += 1
        return seg_id

    def link_text_to_vec(self, text_seg_id: int, vec_seg_id: int) -> None:
        path = self._segment_path(text_seg_id)
        with open(path, "r+b") as f:
            raw = f.read(HEADER_SIZE)
            header = ISHeader.unpack(raw)
            header.next_id = vec_seg_id
            f.seek(0)
            f.write(header.pack())

    def append_text_with_embedding(self, text: str, emb: np.ndarray, quantize: bool = False) -> Tuple[int, int]:
        t_id = self.append_text(text)
        if quantize:
            v_id = self.append_vector_i8(emb, prev_text_id=t_id)
        else:
            v_id = self.append_vector_f32(emb, prev_text_id=t_id)
        self.link_text_to_vec(t_id, v_id)
        return t_id, v_id

    # ---------- read ----------
    def read_text(self, seg_id: int) -> str:
        hdr = self._read_header(seg_id)
        if hdr.data_type != DT_TEXT:
            raise ValueError("Not a text segment")
        return self._read_payload(seg_id, hdr.data_len).decode("utf-8", errors="ignore")

    def read_vector(self, seg_id: int) -> np.ndarray:
        hdr = self._read_header(seg_id)
        buf = self._read_payload(seg_id, hdr.data_len)
        if hdr.data_type == DT_VEC_F32:
            return np.frombuffer(buf, dtype=np.float32, count=hdr.dim)
        if hdr.data_type == DT_VEC_I8:
            q = np.frombuffer(buf, dtype=np.int8, count=hdr.dim).astype(np.float32)
            return q * hdr.scale
        raise ValueError("Not a vector segment")

    def read_coarse(self, seg_id: int) -> bytes:
        hdr = self._read_header(seg_id)
        if hdr.data_type != DT_COARSE:
            raise ValueError("Not a coarse segment")
        return self._read_payload(seg_id, hdr.data_len)

    # ---------- lists ----------
    def list_segments(self, data_type: Optional[int] = None) -> List[int]:
        out = []
        for seg_id in self._glob_segments():
            if data_type is None:
                out.append(seg_id)
            else:
                try:
                    hdr = self._read_header(seg_id)
                    if hdr.data_type == data_type:
                        out.append(seg_id)
                except Exception:
                    pass
        return out

    def list_pairs(self) -> List[Tuple[int, int]]:
        pairs = []
        for tid in self.list_segments(DT_TEXT):
            try:
                th = self._read_header(tid)
            except Exception:
                continue
            if th.next_id >= 0:
                try:
                    eh = self._read_header(th.next_id)
                    if eh.data_type in (DT_VEC_F32, DT_VEC_I8) and eh.prev_id == tid:
                        pairs.append((tid, th.next_id))
                except Exception:
                    pass
        return pairs

    # ---------- search ----------
    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        # Clamp to [-1, 1] to handle floating-point precision issues
        return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))

    def _candidate_ids_from_coarse(self, q: np.ndarray, limit: Optional[int]) -> Optional[List[int]]:
        """
        Optional: decode a simple IVF-like coarse segment to quickly choose candidate eids.
        Format (toy, binary):
          uint32 num_centroids
          for each centroid:
             uint32 centroid_id (eid hint)  ; or bucket id
             uint32 dim
             float32[dim] centroid_vector
        We pick top-N centroid ids as buckets => map to nearby eids via a trivial rule (centroid_id +/- window).
        This is a simple placeholder to satisfy 'special index segments' embodiment.
        """
        coarse_ids = self.list_segments(DT_COARSE)
        if not coarse_ids:
            return None
        # Use the last coarse segment
        blob = self.read_coarse(coarse_ids[-1])
        bio = io.BytesIO(blob)
        try:
            num = struct.unpack("<I", bio.read(4))[0]
        except Exception:
            return None
        scores: List[Tuple[float, int]] = []
        for _ in range(num):
            cid = struct.unpack("<I", bio.read(4))[0]
            cdim = struct.unpack("<I", bio.read(4))[0]
            cvec = np.frombuffer(bio.read(4*cdim), dtype=np.float32, count=cdim)
            s = self._cosine(q, cvec)
            scores.append((s, cid))
        scores.sort(key=lambda x: x[0], reverse=True)
        take = min(len(scores), (limit or 32))
        # Trivial mapping: centroid id is a representative eid; scan a small window around each
        win = 32
        candidates: List[int] = []
        for _, eid_hint in scores[:take]:
            lo = max(0, eid_hint - win)
            hi = eid_hint + win + 1
            # include only vector segments in [lo..hi]
            for eid in range(lo, hi):
                try:
                    hdr = self._read_header(eid)
                    if hdr.data_type in (DT_VEC_F32, DT_VEC_I8):
                        candidates.append(eid)
                except Exception:
                    continue
        # dedup but keep order
        seen = set()
        out: List[int] = []
        for x in candidates:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out or None

    def search_by_vector(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        stride: int = 1,
        coarse_limit: Optional[int] = 16,
    ) -> List[Tuple[float, int, int]]:
        """
        Vector-only scan (with optional coarse prefilter & stride).
        Returns [(score, tid, eid)] sorted by score desc.
        """
        # Optional coarse prefilter to reduce scan
        candidate_eids = self._candidate_ids_from_coarse(query_vec, limit=coarse_limit)
        pairs = self.list_pairs()
        hits: List[Tuple[float, int, int]] = []

        if candidate_eids is not None:
            # Restrict to candidate_eids
            vec_ids = set(candidate_eids)
            iter_pairs = [(tid, eid) for (tid, eid) in pairs if eid in vec_ids]
        else:
            iter_pairs = pairs

        # stride scanning
        if stride > 1:
            iter_pairs = iter_pairs[::max(1, int(stride))]

        for tid, eid in iter_pairs:
            try:
                ev = self.read_vector(eid)
                s = self._cosine(query_vec, ev)
                hits.append((s, tid, eid))
            except Exception:
                continue

        hits.sort(key=lambda x: x[0], reverse=True)
        return hits[:top_k]

    # ---------- stats ----------
    def stat(self) -> Dict[str, int]:
        texts = self.list_segments(DT_TEXT)
        vecs_f = self.list_segments(DT_VEC_F32)
        vecs_q = self.list_segments(DT_VEC_I8)
        return {
            "text_segments": len(texts),
            "vector_segments": len(vecs_f) + len(vecs_q),
            "pairs": len(self.list_pairs()),
            "next_id": self.next_id,
        }
        

    def append_bytes(
        self,
        payload: Union[bytes, str],
        data_type: int = DT_BLOB,
        prev_id: int = -1,
        dim: int = 0,
        scale: float = 1.0,
    ) -> int:
        seg_id = self._next_id
        data = _ensure_bytes(payload)
        hdr = ISHeader(-1, prev_id, len(data), data_type, int(dim), float(scale))
        self._write_segment(seg_id, hdr, data)
        self._next_id += 1
        return seg_id

    def read_bytes(self, seg_id: int) -> bytes:
        hdr = self._read_header(seg_id)
        return self._read_payload(seg_id, hdr.data_len)

    def append_json(self, obj: dict, prev_id: int = -1) -> int:
        import json
        data = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        return self.append_bytes(data, data_type=DT_JSON, prev_id=prev_id)

    def read_json(self, seg_id: int) -> dict:
        import json
        hdr = self._read_header(seg_id)
        if hdr.data_type not in (DT_JSON, DT_TEXT):
            raise ValueError("Not a JSON/text segment")
        raw = self._read_payload(seg_id, hdr.data_len).decode("utf-8", "ignore")
        return json.loads(raw)

    # optional: convenience to store original media “blob” then a JSON/text that points to it
    def append_media_with_manifest(
        self,
        media_bytes: bytes,
        media_kind: str,   # "image" | "audio" | "video" | "pdf" | "blob"
        filename: str | None = None,
        meta: dict | None = None,
    ) -> tuple[int, int]:
        dt = {
            "image": DT_IMAGE,
            "audio": DT_AUDIO,
            "blob":  DT_BLOB,
            "video": DT_BLOB,  # or add a DT_VIDEO if you like
            "pdf":   DT_BLOB,
        }.get(media_kind, DT_BLOB)

        blob_id = self.append_bytes(media_bytes, data_type=dt)
        manifest = {
            "type": media_kind,
            "ref_blob_id": blob_id,
            "filename": filename,
            "meta": meta or {},
        }
        man_id = self.append_json(manifest, prev_id=blob_id)
        return man_id, blob_id

    # ========== Batch operations ==========
    def append_batch(
        self,
        texts: List[str],
        embeddings: List[np.ndarray],
        quantize: bool = False,
    ) -> List[Tuple[int, int]]:
        """
        Batch append multiple text-embedding pairs efficiently.
        
        Args:
            texts: List of text strings
            embeddings: List of embedding vectors
            quantize: Whether to quantize embeddings to int8
            
        Returns:
            List of (text_id, vec_id) tuples
        """
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must have same length")
        
        results = []
        for text, emb in zip(texts, embeddings):
            tid, vid = self.append_text_with_embedding(text, emb, quantize=quantize)
            results.append((tid, vid))
        return results

    def export_to_dict(self, include_vectors: bool = False) -> Dict:
        """
        Export store contents to a dictionary for serialization.
        
        Args:
            include_vectors: Whether to include full vector data (can be large)
            
        Returns:
            Dictionary containing store data
        """
        export = {
            "version": "1.0",
            "dir_path": self.dir_path,
            "next_id": self.next_id,
            "pairs": [],
            "stats": self.stat(),
        }
        
        for text_id, vec_id in self.list_pairs():
            pair_data = {
                "text_id": text_id,
                "vec_id": vec_id,
                "text": self.read_text(text_id),
            }
            
            if include_vectors:
                vec = self.read_vector(vec_id)
                pair_data["vector"] = vec.tolist()
                pair_data["vector_dim"] = len(vec)
            else:
                vec = self.read_vector(vec_id)
                pair_data["vector_dim"] = len(vec)
                pair_data["vector_norm"] = float(np.linalg.norm(vec))
            
            export["pairs"].append(pair_data)
        
        return export

    def delete_segment(self, seg_id: int) -> bool:
        """
        Delete a segment file.
        
        Args:
            seg_id: Segment ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        path = self._segment_path(seg_id)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def compact(self) -> Dict[str, int]:
        """
        Compact store by removing orphaned segments and reindexing.
        Returns statistics about the compaction.
        """
        # Find valid pairs
        valid_pairs = self.list_pairs()
        valid_ids = set()
        for tid, vid in valid_pairs:
            valid_ids.add(tid)
            valid_ids.add(vid)
        
        # Find all segments
        all_ids = set(self._glob_segments())
        
        # Delete orphaned segments
        orphaned = all_ids - valid_ids
        deleted = 0
        for seg_id in orphaned:
            if self.delete_segment(seg_id):
                deleted += 1
        
        return {
            "total_segments": len(all_ids),
            "valid_segments": len(valid_ids),
            "deleted_segments": deleted,
        }

    def append_blob(self, data: bytes) -> int:
        """Convenience method to append a binary blob."""
        return self.append_bytes(data, data_type=DT_BLOB)

    def read_blob(self, seg_id: int) -> bytes:
        """Convenience method to read a binary blob."""
        hdr = self._read_header(seg_id)
        if hdr.data_type != DT_BLOB:
            raise ValueError(f"Not a blob segment (type={hdr.data_type})")
        return self._read_payload(seg_id, hdr.data_len)


# ========== Multi-store (sharding + late fusion) ==========
class MultiStore:
    """
    Holds multiple ISStore shards (directories). Performs per-shard top-k then late fusion.
    """
    def __init__(self, stores: Iterable[ISStore]):
        self.stores = list(stores)

    def search(
        self, q: np.ndarray, per_shard_k: int = 8, final_k: int = 5, stride: int = 1, coarse_limit: Optional[int] = 16
    ) -> List[Tuple[str, float, int, int]]:
        per: List[Tuple[str, float, int, int]] = []
        for s in self.stores:
            hits = s.search_by_vector(q, top_k=per_shard_k, stride=stride, coarse_limit=coarse_limit)
            for (score, tid, eid) in hits:
                per.append((s.dir_path, score, tid, eid))
        per.sort(key=lambda x: x[1], reverse=True)
        return per[:final_k]

# ========== Simple playlist/manifest (HLS-like) ==========
def write_playlist(store: ISStore, path: str) -> None:
    """
    Minimal .m3u8 listing for segments (illustrative embodiment).
    """
    ids = store._glob_segments()
    lines = ["#EXTM3U"]
    for sid in ids:
        fname = f"segment_{sid}.is"
        lines.append("#EXTINF:0.0,")
        lines.append(fname)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
