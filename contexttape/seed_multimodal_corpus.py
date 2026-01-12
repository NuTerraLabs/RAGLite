#!/usr/bin/env python3
"""
seed_multimodal_corpus.py
Creates a mixed-modality sample corpus with RICH content for retrieval.

Run:
  python seed_multimodal_corpus.py --out sample_corpus
  ct ingest-any ./sample_corpus --out-dir wiki_store --quantize --verbose
  ct stat --wiki-dir wiki_store --chat-dir chat_ts
  ct search "photosynthesis light reactions" --wiki-dir wiki_store --chat-dir chat_ts --topk 5 --verbose

Generated files:
- readme.txt (overview with keywords)
- notes.md (longer doc on photosynthesis)
- http_https.md (protocol explainer)
- renaissance_art.md (humanities content)
- kamala_harris.txt (bio/context)
- kb_index.json (structured index with summaries)
- glossary.csv (keyword-to-blurb mapping)
- pixel.png (1x1 PNG), pixel.jpg (1x1 JPG)
- tone.wav (1s, 440Hz mono WAV)
- clip.mp4 (dummy MP4 blob; used to test 'blob' ingest path)

No external dependencies required.
"""

from __future__ import annotations
import os
import base64
import wave
import struct
import math
import argparse
from pathlib import Path

# --- Tiny real 1x1 PNG (transparent) base64 ---
TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB/6X8W8cA"
    "AAAASUVORK5CYII="
)

# --- Tiny 1x1 JPEG (white) base64 (valid) ---
TINY_JPG_B64 = (
    "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAP//////////////////////////////////////////////////////"
    "/////////////////////////////////////////////////////////2wBDAf//////////////////////////"
    "////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAA"
    "AAAAAAAAAAAAAAAAAAb/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFAEBAAAAAAAAAAAAAAAAAAAAAP/EABQRAQAAAAAAA"
    "AAAAAAAAAAAAAAD/2gAMAwEAAhEDEQA/AMQAAAD/2Q=="
)

def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")

def write_bytes_b64(path: Path, b64: str) -> None:
    path.write_bytes(base64.b64decode(b64))

def write_wav_440hz(path: Path, seconds: float = 1.0, sample_rate: int = 16000, amplitude: float = 0.3) -> None:
    """Generate a 1-second 440Hz mono 16-bit PCM WAV using only stdlib."""
    frames = int(seconds * sample_rate)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        for n in range(frames):
            t = n / sample_rate
            sample = int(amplitude * 32767.0 * math.sin(2.0 * math.pi * 440.0 * t))
            wf.writeframes(struct.pack("<h", sample))

def main() -> None:
    ap = argparse.ArgumentParser(description="Seed a rich mixed-modality sample corpus.")
    ap.add_argument("--out", default="sample_corpus", help="Output directory to create/populate")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- README (overview + keywords) ---
    write_text(
        out / "readme.txt",
        (
            "ContextTape Demo Corpus\n"
            "=======================\n"
            "This folder contains a deliberately diverse set of files so your retriever has meaningful\n"
            "context to return. Try queries like:\n"
            "- photosynthesis light-dependent reactions ATP NADPH Calvin cycle\n"
            "- HTTP vs HTTPS TLS handshake status codes idempotent methods\n"
            "- Renaissance art Medici patronage perspective Brunelleschi\n"
            "- Kamala Harris Senate priorities criminal justice immigration healthcare\n"
            "\n"
            "Files:\n"
            "- notes.md                 (science: photosynthesis, with equations)\n"
            "- http_https.md            (networking: protocol explainer)\n"
            "- renaissance_art.md       (humanities: concise overview)\n"
            "- kamala_harris.txt        (bio/context prior to VP term end in Jan 2025)\n"
            "- kb_index.json            (structured summaries for each topic)\n"
            "- glossary.csv             (keyword → short blurb mapping for quick hits)\n"
            "- pixel.png / pixel.jpg    (1x1 images; used to test image ingest path)\n"
            "- tone.wav                 (1s audio tone; tests audio ingest)\n"
            "- clip.mp4                 (dummy blob; tests generic/video path)\n"
        )
    )

    # --- Science: Photosynthesis (longer, queryable) ---
    write_text(
        out / "notes.md",
        (
            "# Photosynthesis — Overview\n\n"
            "Photosynthesis converts light energy into chemical energy stored in glucose. In plants, this occurs in chloroplasts.\n"
            "**Light-dependent reactions** happen in the thylakoid membranes and generate **ATP** and **NADPH** by using sunlight to split water (photolysis), releasing **O₂**.\n\n"
            "**Calvin cycle** (light-independent reactions) occurs in the stroma and fixes **CO₂** into **G3P**, ultimately forming glucose. The simplified overall equation:\n\n"
            "```\n"
            "6 CO₂ + 6 H₂O + light → C₆H₁₂O₆ + 6 O₂\n"
            "```\n\n"
            "Key terms: chlorophyll, photosystems II and I (PSII/PSI), electron transport chain, proton gradient, RuBisCO, carbon fixation.\n"
        )
    )

    # --- Networking: HTTP vs HTTPS ---
    write_text(
        out / "http_https.md",
        (
            "# HTTP vs HTTPS — What’s the difference?\n\n"
            "**HTTP (Hypertext Transfer Protocol)** is an application-layer protocol for the web. It defines methods like **GET**, **POST**, **PUT**, **DELETE** and uses status codes (e.g., **200 OK**, **404 Not Found**, **500 Internal Server Error**).\n\n"
            "**HTTPS** wraps HTTP inside **TLS** to provide encryption, server authentication, and integrity. The **TLS handshake** negotiates keys and ciphers, enabling confidentiality and protecting against on-path attackers.\n\n"
            "Other useful ideas:\n"
            "- **Idempotent** methods: GET, PUT, DELETE (repeated calls should have the same effect).\n"
            "- **Safe** methods: GET, HEAD (shouldn’t change server state).\n"
            "- Persistent connections, HTTP/2 multiplexing, HTTP/3 over QUIC.\n"
        )
    )

    # --- Humanities: Renaissance Art ---
    write_text(
        out / "renaissance_art.md",
        (
            "# Renaissance Art — A brief overview\n\n"
            "The Renaissance (14th–17th centuries) introduced naturalism, perspective, and humanism in European art.\n"
            "Techniques like **linear perspective** (popularized by Brunelleschi) and **chiaroscuro** created depth and realism.\n"
            "Patronage—especially by the **Medici**—fueled the careers of artists such as **Leonardo da Vinci**, **Michelangelo**, and **Raphael**.\n"
            "Subjects included classical mythology, religious narratives, and portraits of civic leaders.\n"
        )
    )

    # --- Bio/context: Kamala Harris (pre-2025 wrap) ---
    write_text(
        out / "kamala_harris.txt",
        (
            "Kamala Harris — Context Snapshot\n"
            "She served as the 49th Vice President of the United States (Jan 20, 2021 – Jan 20, 2025). "
            "Prior roles include U.S. Senator from California (2017–2021) and California Attorney General (2011–2017). "
            "Areas of focus have included criminal justice reform, immigration, healthcare access, and consumer protection.\n"
        )
    )

    # --- Structured index (helps lexical + gives summaries) ---
    write_text(
        out / "kb_index.json",
        (
            "{\n"
            '  "topics": [\n'
            '    {\n'
            '      "id": "photosynthesis",\n'
            '      "title": "Photosynthesis",\n'
            '      "summary": "Light-dependent reactions produce ATP and NADPH; Calvin cycle fixes CO2 into sugars; overall 6CO2+6H2O+light→C6H12O6+6O2."\n'
            "    },\n"
            "    {\n"
            '      "id": "http_vs_https",\n'
            '      "title": "HTTP vs HTTPS",\n'
            '      "summary": "HTTPS adds TLS for encryption and authentication; methods, status codes, and semantics match HTTP."\n'
            "    },\n"
            "    {\n"
            '      "id": "renaissance_art",\n'
            '      "title": "Renaissance Art",\n'
            '      "summary": "Linear perspective, humanism, Medici patronage, and masters like Leonardo, Michelangelo, Raphael."\n'
            "    },\n"
            "    {\n"
            '      "id": "kamala_harris",\n'
            '      "title": "Kamala Harris",\n'
            '      "summary": "Served as U.S. Vice President (2021–2025); previous roles include U.S. Senator and California Attorney General."\n'
            "    }\n"
            "  ]\n"
            "}\n"
        )
    )

    # --- CSV glossary (fast lexical hits) ---
    write_text(
        out / "glossary.csv",
        (
            "keyword,blurb\n"
            "photosynthesis,Converts light to chemical energy via ATP/NADPH and the Calvin cycle.\n"
            "calvin_cycle,CO2 fixation pathway producing G3P; uses ATP and NADPH.\n"
            "tls,Transport Layer Security provides encryption and authentication for HTTPS.\n"
            "status_codes,HTTP responses like 200 OK, 301 Moved Permanently, 404 Not Found.\n"
            "linear_perspective,Geometric method to create depth; codified in the Renaissance.\n"
            "medici,Powerful Florentine patrons who funded major Renaissance works.\n"
            "kamala_harris,Served as U.S. Vice President (2021–2025) and previously Senator/AG.\n"
        )
    )

    # --- Images ---
    write_bytes_b64(out / "pixel.png", TINY_PNG_B64)
    write_bytes_b64(out / "pixel.jpg", TINY_JPG_B64)

    # --- Audio (valid WAV) ---
    write_wav_440hz(out / "tone.wav", seconds=1.0)

    # --- Dummy MP4 (not a real video) ---
    (out / "clip.mp4").write_bytes(b"THIS_IS_NOT_A_REAL_MP4_BUT_OK_FOR_BLOB_DEMO")

    print(f"[OK] Seeded: {out.resolve()}")
    print("Contents:")
    for p in sorted(out.iterdir()):
        print(f" - {p.name} ({p.stat().st_size} bytes)")

    print("\nNext step: ingest into your wiki store, e.g.:")
    print(f"  ct ingest-any {out} --out-dir wiki_store --quantize --verbose")
    print("\nThen check stats:")
    print("  ct stat --wiki-dir wiki_store --chat-dir chat_ts")
    print("\nTry searches (wiki/chat shown separately if you used the updated CLI):")
    print('  ct search "photosynthesis light reactions" --wiki-dir wiki_store --chat-dir chat_ts --topk 5 --verbose')
    print('  ct search "HTTP TLS handshake idempotent methods" --wiki-dir wiki_store --chat-dir chat_ts --topk 5 --verbose')
    print('  ct search "Renaissance art Medici perspective" --wiki-dir wiki_store --chat-dir chat_ts --topk 5 --verbose')

if __name__ == "__main__":
    main()
