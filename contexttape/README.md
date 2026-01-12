# ContextTape 📼

[![PyPI version](https://badge.fury.io/py/contexttape.svg)](https://badge.fury.io/py/contexttape)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/badge/contexttape)](https://pepy.tech/project/contexttape)

**File-Based RAG Made Simple: Zero-infrastructure vector storage for Retrieval-Augmented Generation (RAG)**

Build RAG systems without Pinecone, Weaviate, ChromaDB, or any vector database. Just files.

ContextTape stores text + embeddings in tiny `.is` segment files with 32-byte headers. Copy them anywhere, version in git, ship on USB drives. Works with OpenAI, LangChain, LlamaIndex, or any LLM.

```bash
pip install contexttape
```
## ⚡ 60 Second Start

```bash
# Install
pip install contexttape
export OPENAI_API_KEY="sk-..."

# Create topics.txt
echo "Python_(programming_language)
Machine_learning
Artificial_intelligence" > topics.txt

# Build knowledge base (creates data/wiki/)
ct build-wiki --topics-file topics.txt --limit 3 --verbose

# Search it!
ct search "What is machine learning?"
```

**That's it!** Data is in `data/wiki/segment_*.is`. Copy it anywhere.

👉 **[Full Quick Start Guide →](QUICKSTART.md)**

---
## � Where Does Your Data Go?

**Important:** When you install via pip, the **package code** goes in `site-packages/contexttape/`. Your **data** gets created wherever you specify:

```python
store = ISStore("my_store")        # Creates: ./my_store/ in current directory
store = ISStore("data/knowledge")  # Creates: ./data/knowledge/
```

**👉 [Read WHERE_DATA_GOES.md](WHERE_DATA_GOES.md) for complete details**

---

## �🚀 Quick Start

**Read this first:** [WHAT_YOU_GET.md](WHAT_YOU_GET.md) — Understand what actually gets created  
**Test it now:** `./run_test.sh` — See the system in action  
**Simple guide:** [SIMPLE_GUIDE.md](SIMPLE_GUIDE.md) — Common patterns and usage  

---

## 🚀 Why ContextTape?

| Feature | Traditional Vector DBs | ContextTape |
|---------|----------------------|-------------|
| **Infrastructure** | Server, Docker, cloud service | Just files |
| **Portability** | Database dumps, migrations | `cp -r` / `rsync` |
| **Memory footprint** | GB-scale indexes | ~150MB for 500K tokens |
| **Dependencies** | Heavy (Faiss, PGVector, etc.) | Pure Python + NumPy |
| **Transparency** | Opaque binary indexes | Human-inspectable segments |
| **Cold start** | Index loading time | Instant (mmap) |

### Benchmarks

On a 50-document corpus (487K tokens):
- **Latency**: 48ms median, 98ms p95
- **Throughput**: 18.66 QPS
- **Memory**: <154MB peak RSS
- **Storage**: 4× smaller with int8 quantization

---

## 📦 Installation

```bash
# Basic install
pip install contexttape

# With PDF support
pip install contexttape[pdf]

# With energy monitoring (Intel)
pip install contexttape[energy]

# Everything
pip install contexttape[all]
```

---

## ⚡ Quick Start

### Python API (3 lines to RAG)

```python
from contexttape import ISStore, get_client, embed_text_1536

# Create a store and embed some docs
store = ISStore("my_knowledge")
client = get_client()  # requires OPENAI_API_KEY

docs = ["Neural networks are universal function approximators.",
        "Fire behavior depends on fuel, weather, and terrain.",
        "Quantum computing uses superposition and entanglement."]

for doc in docs:
    vec = embed_text_1536(client, doc)
    store.append_text_with_embedding(doc, vec)

# Search
query_vec = embed_text_1536(client, "how does quantum computing work?")
for score, text_id, vec_id in store.search_by_vector(query_vec, top_k=3):
    print(f"{score:.3f}: {store.read_text(text_id)[:80]}...")
```

### CLI

```bash
# Set your API key
export OPENAI_API_KEY="sk-..."

# Ingest documents
ct ingest-path ./docs --out-dir my_store --exts md txt pdf

# Search
ct search "machine learning optimization" --topk 5

# Chat with retrieval
ct chat --topk 5 --verbose
```

---

## 🧱 The `.is` Segment Format

Each item is a single file:

```
[ 32-byte header ][ payload bytes ]

Header (little-endian):
├─ int32  next_id      # Link to paired segment
├─ int32  prev_id      # Back-link
├─ int32  data_len     # Payload size
├─ int32  data_type    # 0=text, 1=vec_f32, 2=vec_i8
├─ int32  dim          # Vector dimension (0 for text)
├─ float32 scale       # Quantization scale
└─ 8 bytes reserved    # Timestamp
```

**Text segment** (`data_type=0`): UTF-8 payload, `next_id` → its vector  
**Vector segment** (`data_type=1|2`): float32/int8 array, `prev_id` → its text

---

## 🔥 Key Features

- **🗃️ No Vector DB** — Storage uses ordinary files, not Faiss/Milvus/PGVector
- **📐 32-byte Headers** — Minimal metadata linking text ↔ vectors
- **⚡ Int8 Quantization** — 4× smaller with `--quantize` flag
- **🔄 Late Dereference** — Read text only for top-k hits
- **🌍 Multi-Store Fusion** — Merge results from multiple directories
- **📊 Hybrid Reranking** — Vector similarity + lexical overlap
- **🔒 Append-Only** — Crash-safe, easy snapshotting
- **🪫 Energy-Aware Mode** — Reduce compute under power limits

---

## 📖 Documentation

### Core Documentation

- **[ORGANIZATION.md](ORGANIZATION.md)** — 🗂️ Project structure and directory layout
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** — 📋 Detailed file organization
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** — ⚡ Quick reference card
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — 🤝 Contribution guide
- **[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)** — 📜 Community standards

### Examples & Tutorials

- **[examples/quickstart.py](examples/quickstart.py)** — 7 basic examples
- **[examples/tutorial.py](examples/tutorial.py)** — 5 step-by-step tutorials
- **[examples/advanced_usage.py](examples/advanced_usage.py)** — 7 advanced patterns

### API Documentation

- **[docs/api_reference.md](docs/api_reference.md)** — Complete API reference
- **[docs/architecture.md](docs/architecture.md)** — System architecture
- **[docs/performance.md](docs/performance.md)** — Performance guide

### 🧹 Cleaning Up

When you run examples, they create temporary directories (`*_store/`, `*_ts/`). These are **user data**, not source code:

```bash
bash cleanup_stores.sh  # Remove all temporary stores
```

See [ORGANIZATION.md](ORGANIZATION.md) for details.

---

## 🗂️ Table of contents

- [Why ContextTape](#why-contexttape)
- [Architecture](#architecture)
- [Quick start](#quick-start)
- [Installation](#installation)
- [CLI usage](#cli-usage)
  - [Build from wiki topics (showcase)](#build-from-wiki-topics-showcase)
  - [Ingest any path (generic)](#ingest-any-path-generic)
  - [Search](#search)
  - [Chat](#chat)
  - [Stats and maintenance](#stats-and-maintenance)
- [Programmatic API](#programmatic-api)
- [Storage format](#storage-format)
- [Configuration](#configuration)
- [Performance notes](#performance-notes)
- [Troubleshooting](#troubleshooting)
- [Security and privacy](#security-and-privacy)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Why ContextTape

- **Bring-your-own-files.** Point it at any folder. Ingest text-like files out of the box; PDFs optionally via `pypdf`.
- **No vector DB required.** Pairs of text and embeddings are stored as tiny `.ts` files on disk with minimal headers.
- **Deterministic, transparent storage.** Every item is visible and inspectable as a file.
- **Composable.** Works as a CLI and as a Python library; plug it into your own agents and services.
- **Lightweight retrieval.** Cosine similarity for candidates, lexical overlap re-rank, and simple diversity heuristics.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                        CLI (ct)                          │
│  build-wiki | ingest-path | search | chat | stat    │
└──────────────┬──────────────────────────┬────────────────┘
               │                          │
               ▼                          ▼
        Ingestion pipelines             Query layer
    (wiki, folder walker)        (embeddings + search + rerank)
               │                          │
               └───────────────┬──────────┘
                               ▼
                       Storage subsystem
                    (ISStore: .is segment files)

Each .is file:
[ 32 bytes header ][ payload ]
- Text segment: UTF-8 text payload
- Vector segment: float32 embedding payload
- Headers link text <-> vector (next/prev IDs)
```

---

## Quick start

```bash
# 1) Create and activate a virtual environment (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate

# 2) Install
pip install contexttape
# or with PDF extraction
pip install contexttape[pdf]

# 3) Set your OpenAI API key (required for embeddings/chat)
export OPENAI_API_KEY="sk-..."

# 4) Build the small wiki showcase (50 topics)
ct build-wiki --topics-file scripts/topics.example.txt --out-dir wiki_store --limit 50 --verbose

# 5) Search the store
ct search "quantum computing applications" --topk 5

# 6) Chat with retrieval
ct chat --topk 5 --verbose
```

To ingest your own files:

```bash
# Ingest everything under ./docs (text-like files + PDFs if pypdf installed)
ct ingest-path ./docs --out-dir my_store --exts md txt pdf --max-pdf-pages 15 --verbose
```

---

## Installation

Requirements:
- Python 3.9+
- `openai` (for embeddings/chat)
- `requests`, `beautifulsoup4` (for the wiki showcase only)
- Optional: `pypdf` for PDF extraction

Install:

```bash
pip install contexttape
# or with optional dependencies
pip install contexttape[pdf]
```

Console entry point is `ct`.

---

## CLI usage

Run `ct --help` for a full list of commands.

### Build from wiki topics (showcase)

Takes titles from a text file (one title per line), fetches the pages, strips markup, embeds text (`text-embedding-3-small`), and writes text + embedding pairs into `.ts` files.

```bash
ct build-wiki \
  --topics-file scripts/topics.example.txt \
  --out-dir wiki_store \
  --limit 50 \
  --verbose
```

- Output: `wiki_store/segment_*.is`
- Each title yields two segments: one text `.is`, one vector `.is` linked via header IDs.

### Ingest any path (generic)

Walks a folder or single file and ingests supported types. Best-effort decoding for text-like files; optional PDF support.

```bash
ct ingest-path ./my_docs \
  --out-dir my_store \
  --exts md txt pdf \
  --max-bytes 1048576 \
  --max-pdf-pages 20 \
  --verbose
```

Notes:
- If `--exts` is omitted, a default set of text/code/log extensions plus `.pdf` is used.
- PDFs are read only when `pypdf` is installed; otherwise PDFs are skipped.
- Use `--max-bytes` to cap the size per text file, and `--max-pdf-pages` to cap PDF pages.

### Search

Searches across the default stores (`wiki_store` via `WIKI_IS_DIR`, and `chat_ts` via `CHAT_IS_DIR`) with embedding similarity, re-ranks using lexical overlap, and prints top results with file paths.

```bash
ct search "fire propagation modeling" --topk 5
```

You’ll see something like:

```
[TOP 1] src=wiki score=0.4321 text_seg=12 emb_seg=13
  text_path=wiki_store/segment_12.is
  vec_path= wiki_store/segment_13.is
  preview: Fire propagation is influenced by fuel, weather, and topography ...
```

### Chat

Retrieval-augmented chat that uses the selected context blocks (from wiki + chat stores) to form a prompt and calls a chat model.

```bash
ct chat --topk 5 --verbose
```

Special questions:
- “what was my first question” returns the earliest stored user message from the chat store (`chat_ts`), purely from `.ts` history.

### Stats and maintenance

```bash
# Show counts, first headers, next_id, etc.
ct stat

# Reset chat store only (does not touch wiki or custom stores)
ct reset-chat
```

---

## Programmatic API

Minimal example for building a store and searching.

```python
from contexttape.storage import ISStore
from contexttape.embed import get_client, embed_text_1536
from contexttape.search import combined_search

# Build a store from in-memory strings
store = ISStore("example_store")
client = get_client()

docs = [
    "Neural networks are function approximators composed of layers.",
    "Fire behavior depends on fuel, weather, and terrain.",
    "Quantum computing leverages superposition and entanglement.",
]
for d in docs:
    vec = embed_text_1536(client, d)
    store.append_text_with_embedding(d, vec)

# Search the same store (as both wiki and chat, for simplicity)
q = "how does quantum entanglement help computation"
qvec = embed_text_1536(client, q)
hits = combined_search(q, qvec, wiki_store=store, chat_store=store, top_k=3)

for src, score, tid, eid in hits:
    text = store.read_text(tid)
    print(src, score, tid, eid, text[:120])
```

---

## Storage format

Each item is a single `.ts` file:

```
[ 32-byte header ][ payload bytes ]
```

Header layout (little-endian, 32 bytes total):

```
int32 next_id
int32 prev_id
int32 data_len
int32 data_type   # 0 = text, 1 = vector_f32
int32 dim         # vector dimension, 0 for text
float32 scale     # reserved
8 bytes reserved
```

- **Text segment**: `data_type=0`, `payload = UTF-8 bytes`. `next_id` points to its vector segment.
- **Vector segment**: `data_type=1`, `payload = float32[dim]` in row-major order. `prev_id` points back to its text segment.

By convention, `ISStore.append_text_with_embedding` writes a text segment, then a vector segment, and links them in headers.

---

## Configuration

Environment variables:

- `OPENAI_API_KEY` – required for embeddings and chat.
- `WIKI_IS_DIR` – default wiki store directory (default: `wiki_real_ts` or your chosen path).
- `CHAT_IS_DIR` – default chat store directory (default: `chat_ts`).
- `TOP_K` – default top-k in CLI search/chat (default: `5`).
- `DEBUG_DIR` – where verbose prompt/context logs are written when `--verbose` is used (default: `debug`).

You can also override store paths at runtime in the CLI with `--out-dir` for ingestion commands.

---

## Performance notes

- **Chunking**: The generic ingester reads whole files up to `--max-bytes`. For very large files, consider pre-chunking by paragraphs or sections before ingestion (roadmap includes first-class chunkers).
- **Embeddings**: `text-embedding-3-small` is used by default (1536-d). Swap to a different model in `embed.py` if needed.
- **Re-ranking**: After cosine candidate generation, lexical overlap on normalized tokens boosts topical precision. Adjust weights in `search.py` if you need a different balance.
- **Parallel ingestion**: Current reference CLI is single-process by default for simplicity. You can parallelize ingestion in your own wrapper or extend the CLI.
- **Disk**: `.ts` files are extremely small for short texts. Large corpora will produce many files; on Linux/macOS this is generally fine, but you can shard into subfolders if you prefer (roadmap).

---

## Troubleshooting

- `ImportError: cannot import name 'iter_files'`: Ensure `src/contexttape/ingest_generic.py` exists and you reinstalled the package with `pip install -e .`.
- `ct: command not found`: Verify the venv is active and `pip install -e .` succeeded. Reopen your shell or re-activate the venv.
- PDF text extraction is empty: Install optional extra `pip install -e .[pdf]` and try again. Some PDFs have no extractable text; OCR is out of scope for the reference implementation.
- OpenAI errors: Ensure `OPENAI_API_KEY` is set and the model names used in `embed.py` and `cli.py` exist for your account.

---

## Security and privacy

- ContextTape writes your source texts to disk as UTF-8 text segments alongside their embeddings. Treat the store directory as sensitive if your sources are not public.
- The default reference implementation sends content to OpenAI for embeddings and chat. If you require air-gapped or self-hosted embeddings, you can swap out `embed.py` to call your own embedding model and keep the rest of the system unchanged.

---

Commands overview
1) ct build-wiki

Fetches wiki pages by title and ingests as text+vector pairs.

Usage

ct build-wiki \
  --topics-file scripts/topics.example.txt \
  --out-dir wiki_store \
  --limit 50 \
  --min-chars 800 \
  --verbose


Flags

--topics-file (required): Text file with one title per line.

--out-dir: Target store for the pages (default $WIKI_IS_DIR).

--limit: Max number of titles to ingest.

--min-chars: Skip very short pages.

--skip-fences: Skip lines starting with ``` during topic loading (defaults to on).

--verbose: Print progress.

2) ct ingest-path

Generic folder/file ingester for text-like files (+ optional PDFs).

Usage

# Ingest specific extensions
ct ingest-path ./docs \
  --out-dir wiki_store \
  --exts md txt pdf \
  --max-bytes 1048576 \
  --max-pdf-pages 20 \
  --follow-symlinks \
  --verbose


Flags

path (positional): File or directory.

--out-dir: Target store (default $WIKI_IS_DIR).

--exts: One or more extensions (without dots or with) to include.

--max-bytes: Per-file read cap for text files.

--max-pdf-pages: Pages to parse per PDF (requires PyMuPDF/pypdf depending on your impl).

--follow-symlinks: Traverse symlinks.

--verbose: Print progress.

Tip: If you need images/audio/video/blobs, use ingest-any (below), not ingest-path.

3) ct ingest-any

Best-effort ingestion of mixed modalities (text, image, audio, pdf, video, other blobs).

Usage

ct ingest-any ./sample_corpus \
  --out-dir wiki_store \
  --quantize \
  --max-bytes 1048576 \
  --verbose


Flags

path (positional): File or directory.

--out-dir: Target store (default $WIKI_IS_DIR).

--quantize: Store vectors as int8 (space-saving).

--max-bytes: Fallback read cap for arbitrary binary if needed.

--verbose: Print progress.

What it does

Text (.txt,.md,.json,.yaml,.csv,.tsv,.log…): embeds the content.

Images (.png,.jpg,...): embeds the pixels (via embed_image) and stores a JSON manifest (caption + optional blob reference).

Audio (.wav,.mp3,...): embeds the audio (via embed_audio) and stores a JSON manifest (+ optional blob).

PDF: extracts text (when supported), embeds it, stores a JSON manifest (+ optional blob).

Video (.mp4,.mov,...): default = a filename summary embedded (you can later extend to keyframes).

Other: stored as “blob” with filename summary embedded.

4) ct search

Searches your stores and prints per-store results (wiki and chat) separately.

Usage

ct search "photosynthesis light reaction" \
  --topk 8 \
  --wiki-dir wiki_store --chat-dir chat_ts \
  --verbose


Flags

query (positional): Your search text.

--topk: Candidate count per fused pass (default $TOP_K).

--wiki-topk: Override per-store top-k shown for WIKI section.

--chat-topk: Override per-store top-k shown for CHAT section.

--wiki-dir, --chat-dir: Override store locations.

--energy-aware: Auto-tune k/stride to save energy (see below).

--max-power-budget: Power cap hint for energy-aware mode (Watts).

--verbose: Also prints file paths and previews.

What you’ll see

=== WIKI RESULTS === … top items from the wiki store only.

=== CHAT RESULTS === … top items from the chat store only.

5) ct chat

Retrieval-augmented chat. It:

searches both stores,

selects relevant blocks with thresholds,

builds a prompt that includes labeled sections:

USER CHAT (recent turns),

KNOWLEDGE (retrieved context, split by WIKI vs CHAT),

WIKI (unfused top-k) and CHAT (unfused top-k) as safety nets,

calls the chat model,

appends the turn to the chat store.

Typical usage

ct chat \
  --wiki-dir wiki_store --chat-dir chat_ts \
  --topk 8 \
  --alpha 0.55 \
  --min-score 0.32 \
  --min-lex 0.12 \
  --min-hybrid 0.28 \
  --max-context-blocks 5 \
  --verbose


Important knobs (what they mean & how to set them)

--topk
How many candidate chunks to consider from the fused search (hybrid scoring) before relevance gating.

Start: 8

Raise (12–20) for broader retrieval; lower (3–5) for speed.

--alpha
Hybrid scoring weight between vector similarity and lexical overlap:

hybrid
=
𝛼
⋅
vector_sim
  
+
  
(
1
−
𝛼
)
⋅
lex_overlap
hybrid=α⋅vector_sim+(1−α)⋅lex_overlap

1.0 = pure vector; 0.0 = pure lexical.

Start around 0.5–0.7.

More abstract/semantic queries → push up (0.65–0.8).

Precise keyword queries → push down (0.3–0.5).

--min-score
Minimum vector similarity to keep a block during gating.

Start: 0.30–0.35.

If you’re missing relevant blocks, lower it slightly. If you see noise, raise it.

--min-lex
Minimum lexical overlap to keep a block. Helps ensure term alignment.

Start: 0.10–0.15.

Raise if you want stricter keyword alignment; lower if phrasing varies a lot.

--min-hybrid
Minimum final score after hybrid mix to keep a block.

Start: 0.25–0.30.

This is the final gate—if relevant content isn’t making it in, reduce this slightly.

--max-context-blocks
Hard cap on how many retrieved blocks will be put into the prompt.

Start: 5.

If answers feel under-informed, try 6–8 (watch token cost).

--energy-aware, --max-power-budget
Enables an adaptive policy that lowers topk and increases stride (skips some segments) when under power pressure.

Useful on laptops/servers to save watts.

E.g. --energy-aware --max-power-budget 12.

Reading the verbose output

=== WIKI (unfused top-k) === / === CHAT (unfused top-k) ===
These are per-store quick scans (no fusion) shown to the model as a backstop, ensuring the assistant can “see” the best pure-vector matches from each store even if the fused/gated list got too strict.

===== NO KNOWLEDGE SELECTED (or filtered) =====
Means the gates removed everything (often because the prompt was small talk like “hi”). Lower thresholds if this happens on real questions.

Pro tips

If you reset the chat store, your first turns won’t have chat memory yet; that’s normal.

If a WIKI block appears unrelated, check thresholds; sometimes vector sim is high but lex is low—tighten --min-lex or raise --min-hybrid.

6) ct stat

Counts segments/pairs and computes an exact token count by re-encoding stored texts.

Usage

ct stat --wiki-dir wiki_store --chat-dir chat_ts


Output

Number of text/vector segments, pairs, next_id.

Total tokens (useful to estimate embedding/storage cost).

7) ct bench

Micro-benchmark: size, tokens, latency, QPS, memory, and energy hints.

Usage

ct bench \
  --wiki-dir wiki_store --chat-dir chat_ts \
  --queries-file /tmp/queries.txt \
  --max-queries 50 \
  --repeats 5 \
  --topk 8 \
  --energy-aware \
  --max-power-budget 15 \
  --assume-power-watts 15 \
  --out-json bench.json \
  --out-csv bench.csv \
  --out-md bench.md \
  --verbose


Flags

--queries-file: Newline-separated queries (fallback to built-ins if omitted).

--max-queries: Cap how many queries to use from the file.

--repeats: Number of passes over the set (stabilizes stats).

--topk, --energy-aware, --max-power-budget: Same behavior as chat/search.

--assume-power-watts: If hardware energy meters are missing, estimate energy as Watts × elapsed_time.

--out-json|--out-csv|--out-md: Save a structured report(s).

Report fields (high value)

Latency (wall_ms_p50, p95, mean), qps_mean

Memory (rss_mb_max, pss_mb_max if available)

Energy backend + Joules (pkg/dram if available or assumed)

Store sizes (MB), tokens, pairs

8) ct reset-chat

Clears the chat store directory (does not touch wiki).

Usage

ct reset-chat --chat-dir chat_ts

Tuning guide (quick recipes)

A) Exact-ish keyword Q&A over your docs

ct chat \
  --alpha 0.4 \
  --min-lex 0.18 \
  --min-hybrid 0.32 \
  --topk 10 \
  --max-context-blocks 6 \
  --verbose


More weight on lexical; slightly stricter hybrid minimum.

B) Conceptual / semantic queries (looser phrasing)

ct chat \
  --alpha 0.7 \
  --min-score 0.30 \
  --min-lex 0.10 \
  --min-hybrid 0.26 \
  --topk 12 \
  --max-context-blocks 6 \
  --verbose


More vector weighting; easier lex gate so synonyms survive.

C) Battery-saver / constrained

ct chat \
  --topk 6 \
  --max-context-blocks 4 \
  --energy-aware \
  --max-power-budget 12 \
  --verbose

Common pitfalls & fixes

“No knowledge selected” even for real questions

Lower --min-hybrid a bit, e.g. 0.28 → 0.24

Lower --min-lex (e.g. 0.15 → 0.10)

Raise --topk (e.g. 8 → 12)

Irrelevant wiki blocks show up

Raise --min-lex and/or --min-hybrid

Lower --alpha if your queries rely on precise keywords

Images/audio show in search but not meaningful

That’s expected unless you add captioning/transcription. The ingest-any manifest helps, but for rich semantics, add real image captioning / ASR and embed that text too.

ct not found / stale code

Reinstall after edits: pip install -e .

Reactivate venv or open a new shell.

Custom file extension

Set CTX_SEG_EXT=.custom to write/read segment_*.custom (default: .is).

Debugging: what verbose mode writes

When you run with --verbose, we also write:

debug/hits.json – raw hits (source, score, tid, eid)

debug/context.md – the exact retrieved blocks sent to the model

debug/prompt.txt – the full prompt used

Use these to tune thresholds and verify per-store sections are included.

One last sanity check

If you want to force clear separation in outputs when testing:

ct search "kamala harris" \
  --wiki-dir wiki_store --chat-dir chat_ts \
  --topk 8 --verbose


You should see two blocks in the output:

=== WIKI RESULTS ===
...only wiki...
=== CHAT RESULTS ===
...only chat...


And in ct chat --verbose, you’ll see:

=== KNOWLEDGE (retrieved context) === split into WIKI CONTEXT and CHAT MEMORY, plus

=== WIKI (unfused top-k) === and === CHAT (unfused top-k) ===.

## Roadmap

- Pluggable chunkers (headings, paragraphs, code-aware).
- Pluggable rerankers (BM25, LLM-based, hybrid scoring).
- Named stores: query across multiple stores with selection rules.
- Streaming ingestion and background workers.
- Optional sharded store layout for very large corpora.
- Optional FAISS/HNSW index builder for faster approximate nearest neighbor search.

---

## Contributing

1. Fork the repository and create a branch.
2. Use `ruff` or `black` for formatting (optional).
3. Add tests if you alter core storage or search behavior.
4. Open a PR and describe the change, rationale, and any compatibility notes.

---

## License

Licensed under the MIT License. See `LICENSE` for details.

---

