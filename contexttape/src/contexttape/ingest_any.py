# ingest_any.py
from __future__ import annotations
import os, io, mimetypes, json
from typing import Optional, Tuple
import numpy as np

from .storage import ISStore, DT_TEXT
from .embed import get_client, embed_text_1536, embed_image, embed_audio

TEXT_EXTS  = {".txt", ".md", ".rst", ".json", ".yaml", ".yml", ".csv", ".tsv", ".log"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac"}
PDF_EXTS   = {".pdf"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

def _read_bytes(path: str, max_bytes: Optional[int] = None) -> bytes:
    with open(path, "rb") as f:
        data = f.read(max_bytes or -1)
    return data

def _read_text_utf8(path: str, max_bytes: Optional[int] = None) -> str:
    raw = _read_bytes(path, max_bytes=max_bytes)
    return raw.decode("utf-8", errors="ignore")

def _summarize_text_for_payload(text: str, max_chars: int = 2000) -> str:
    s = text.strip()
    if len(s) > max_chars:
        s = s[:max_chars] + " …"
    return s

def _caption_for_image(path: str) -> str:
    base = os.path.basename(path)
    return f"Image file: {base}"

def _summary_for_audio(path: str, duration_sec: Optional[float] = None) -> str:
    base = os.path.basename(path)
    dur = f"{duration_sec:.1f}s" if duration_sec else "unknown duration"
    return f"Audio file: {base} ({dur})"

def _summary_for_pdf(path: str, pages: Optional[int]) -> str:
    base = os.path.basename(path)
    p = f"{pages} pages" if pages is not None else "unknown pages"
    return f"PDF document: {base} ({p})"

def _summary_for_video(path: str) -> str:
    base = os.path.basename(path)
    return f"Video file: {base}"

def _build_text_payload(manifest: dict) -> str:
    return json.dumps(manifest, ensure_ascii=False, indent=2)

def ingest_text(store: ISStore, path: str, quantize_vec: bool = False, max_bytes: Optional[int] = None) -> Tuple[int,int]:
    client = get_client()
    text = _read_text_utf8(path, max_bytes=max_bytes)
    summary = _summarize_text_for_payload(text)
    emb = embed_text_1536(client, summary, verbose=False)
    payload = _build_text_payload({
        "type": "text",
        "filename": os.path.basename(path),
        "preview": summary,
    })
    tid, vid = store.append_text_with_embedding(payload, emb, quantize=quantize_vec)
    return tid, vid

def ingest_image(store: ISStore, path: str, also_store_blob: bool = True, quantize_vec: bool = False) -> Tuple[int,int]:
    b = _read_bytes(path)
    # Assuming embed_image(bytes) -> np.ndarray (no client needed). If your impl needs a client, do: embed_image(get_client(), b)
    emb = embed_image(b)
    caption = _caption_for_image(path)
    manifest = {
        "type": "image",
        "filename": os.path.basename(path),
        "caption": caption,
        "blob_ref": None,
    }
    if also_store_blob:
        man_id, blob_id = store.append_media_with_manifest(b, "image", filename=os.path.basename(path))
        manifest["blob_ref"] = {"manifest_id": man_id, "blob_id": blob_id}

    payload = _build_text_payload(manifest)
    # Also embed the caption text so retrieval works even if image embedding isn’t searched
    emb_text = embed_text_1536(get_client(), caption, verbose=False)
    tid, vid = store.append_text_with_embedding(payload, emb_text, quantize=quantize_vec)
    return tid, vid

def ingest_audio(store: ISStore, path: str, also_store_blob: bool = True, quantize_vec: bool = False) -> Tuple[int,int]:
    b = _read_bytes(path)
    # Assuming embed_audio(bytes) -> np.ndarray. If needs client: embed_audio(get_client(), b)
    emb = embed_audio(b)
    summary = _summary_for_audio(path, duration_sec=None)
    manifest = {
        "type": "audio",
        "filename": os.path.basename(path),
        "summary": summary,
        "blob_ref": None,
    }
    if also_store_blob:
        man_id, blob_id = store.append_media_with_manifest(b, "audio", filename=os.path.basename(path))
        manifest["blob_ref"] = {"manifest_id": man_id, "blob_id": blob_id}

    payload = _build_text_payload(manifest)
    emb_text = embed_text_1536(get_client(), summary, verbose=False)
    tid, vid = store.append_text_with_embedding(payload, emb_text, quantize=quantize_vec)
    return tid, vid

def ingest_pdf(store: ISStore, path: str, also_store_blob: bool = True, quantize_vec: bool = False, max_pages: int = 12) -> Tuple[int,int]:
    client = get_client()
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        pages = min(len(doc), max_pages)
        text_parts = []
        for i in range(pages):
            text_parts.append(doc[i].get_text("text"))
        doc.close()
        full_text = "\n".join(text_parts).strip() or "(no extractable text)"
    except Exception:
        full_text = "(text extraction unavailable)"
        pages = None

    summary = _summary_for_pdf(path, pages)
    emb_source = full_text if full_text and full_text != "(text extraction unavailable)" else summary
    emb = embed_text_1536(client, emb_source, verbose=False)

    manifest = {
        "type": "pdf",
        "filename": os.path.basename(path),
        "summary": summary,
        "text_preview": _summarize_text_for_payload(emb_source, 1500),
        "blob_ref": None,
    }
    if also_store_blob:
        b = _read_bytes(path)
        man_id, blob_id = store.append_media_with_manifest(b, "pdf", filename=os.path.basename(path))
        manifest["blob_ref"] = {"manifest_id": man_id, "blob_id": blob_id}

    payload = _build_text_payload(manifest)
    tid, vid = store.append_text_with_embedding(payload, emb, quantize=quantize_vec)
    return tid, vid

def ingest_video(store: ISStore, path: str, also_store_blob: bool = True, quantize_vec: bool = False) -> Tuple[int,int]:
    client = get_client()
    summary = _summary_for_video(path)
    emb = embed_text_1536(client, summary, verbose=False)
    manifest = {
        "type": "video",
        "filename": os.path.basename(path),
        "summary": summary,
        "blob_ref": None,
    }
    if also_store_blob:
        b = _read_bytes(path)
        man_id, blob_id = store.append_media_with_manifest(b, "video", filename=os.path.basename(path))
        manifest["blob_ref"] = {"manifest_id": man_id, "blob_id": blob_id}

    payload = _build_text_payload(manifest)
    tid, vid = store.append_text_with_embedding(payload, emb, quantize=quantize_vec)
    return tid, vid

def ingest_any(store: ISStore, path: str, quantize_vec: bool = False, max_bytes: Optional[int] = None) -> Tuple[int,int]:
    ext = os.path.splitext(path)[1].lower()
    if ext in TEXT_EXTS:
        return ingest_text(store, path, quantize_vec=quantize_vec, max_bytes=max_bytes)
    if ext in IMAGE_EXTS:
        return ingest_image(store, path, quantize_vec=quantize_vec)
    if ext in AUDIO_EXTS:
        return ingest_audio(store, path, quantize_vec=quantize_vec)
    if ext in PDF_EXTS:
        return ingest_pdf(store, path, quantize_vec=quantize_vec)
    if ext in VIDEO_EXTS:
        return ingest_video(store, path, quantize_vec=quantize_vec)

    # Fallback: generic blob summarized via filename and embedded via text model
    client = get_client()
    b = _read_bytes(path, max_bytes=max_bytes)
    base = os.path.basename(path)
    summary = f"File: {base}"
    emb = embed_text_1536(client, summary, verbose=False)
    man_id, blob_id = store.append_media_with_manifest(b, "blob", filename=base)
    payload = _build_text_payload({
        "type": "blob",
        "filename": base,
        "summary": summary,
        "blob_ref": {"manifest_id": man_id, "blob_id": blob_id},
    })
    return store.append_text_with_embedding(payload, emb, quantize=quantize_vec)
