# RAGLite Repository

> **Note:** This repository contains multiple projects. The main production-ready package is **ContextTape** (located in the `contexttape/` directory).

## 📦 Main Package: ContextTape

**👉 [Go to ContextTape Package →](contexttape/README.md)**

The production-ready, open-source RAG storage system is in:
```
contexttape/                    ← THE MAIN PACKAGE
├── src/contexttape/           ← Source code
├── tests/                     ← Test suite (55 tests)
├── examples/                  ← Usage examples
├── docs/                      ← Documentation
└── README.md                  ← Full package docs
```

**Quick links:**
- 📖 [Package Documentation](contexttape/README.md)
- 🗂️ [Project Structure Guide](contexttape/PROJECT_STRUCTURE.md)
- 🚀 [Quick Reference](contexttape/QUICK_REFERENCE.md)
- 🤝 [Contributing Guide](contexttape/CONTRIBUTING.md)

## 🧪 Other Projects in This Repository

| Directory | Status | Purpose |
|-----------|--------|---------|
| `contexttape/` | ✅ **Production** | Main RAG package |
| `cleanup/` | 🧪 Experimental | Data cleanup utilities |
| `newdbtype/`, `newrag/` | 🧪 Experimental | Database/RAG experiments |
| `runner.py` | 📝 Legacy | Old runner scripts |

---

# 🧠 ContextTape — Overview
### Persistent Vector Memory for Retrieval-Augmented Generation (RAG-Light, Database-Free)

[![Tests](https://github.com/NuTerraLabs/contexttape/actions/workflows/ci.yml/badge.svg)](https://github.com/NuTerraLabs/contexttape/actions)
[![PyPI](https://img.shields.io/pypi/v/contexttape)](https://pypi.org/project/contexttape/)
[![Python](https://img.shields.io/pypi/pyversions/contexttape)](https://pypi.org/project/contexttape/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 🔍 What is ContextTape?

**ContextTape** is a **database-free retrieval architecture** that replaces vector databases with a **pure file-segment system**.

Each content item is stored as two paired files:
- A **text segment** (`segment_T.is`) containing UTF-8 text.
- A **vector segment** (`segment_E.is`) containing an embedding.

Each segment begins with a **fixed 32-byte header** that encodes the metadata needed to link, identify, and retrieve content.

At query time, the system:
1. Embeds the query into a vector.
2. **Scans only vector segments** sequentially (no ANN index).
3. Maintains a **top-k heap** of best matches.
4. **Late-dereferences** corresponding text segments only for top results.
5. Optionally applies **hybrid re-ranking** (vector + lexical/domain).
6. Assembles the retrieval context for an LLM or other downstream use.

This design shifts the bottleneck from compute-bound vector search to **sequential I/O**, drastically reducing memory use and power draw.

---

## ⚙️ Key Features

- 🧩 **No Vector DB** — storage uses ordinary files, not Faiss/PGVector/Milvus/etc.
- 🧱 **Segment Headers** — 32B metadata block linking text and vectors.
- ⚡ **Quantization** — int8 vectors with per-segment scale → 4× smaller.
- 🔄 **Late Dereference** — read text only for top-k hits.
- 🧮 **Stride Scanning** — skip segments to trade accuracy for speed.
- 🌍 **Multi-Store Fusion** — merge results from multiple directories.
- 🧠 **Coarse Prefilter** — optional lightweight centroid filtering.
- 🔒 **Append-Only Writes** — crash-safe, easy snapshotting.
- 🧾 **Auditability** — exact bytes passed to the model are reproducible.
- 🪫 **Energy-Aware Mode** — reduce top-k or stride under power limits.
- 🖼️ **Visual Container Option** — (future) deterministic vector-in-frame mapping.

---

## 📁 Directory Layout

```text
src/contexttape/
  ├── storage.py        # File-segment store (this README documents it)
  ├── cli.py            # CLI: ingest, search, chat, bench, stat, reset
  ├── embed.py          # Embedding utilities
  ├── ingest_*          # Chunkers for text/wiki/etc
  ├── energy.py         # Power-aware tuning (optional)
  └── search.py         # Hybrid vector/lexical rerank
wiki_store/             # Example corpus store
chat_ts/                # Chat memory store
bench/                  # Benchmark outputs
dist/                   # Exported playlists/manifests
```

---

## 🚀 Quick Start

### Installation

```bash
pip install contexttape
```

### Basic Usage

```python
from contexttape import TSStore
import numpy as np

# Create a store
store = TSStore("my_knowledge_base")

# Add documents with embeddings
text = "Machine learning is transforming AI."
embedding = np.random.randn(1536).astype(np.float32)  # Use real embeddings in production
text_id, vec_id = store.append_text_with_embedding(text, embedding, quantize=True)

# Search
query_embedding = np.random.randn(1536).astype(np.float32)
results = store.search_by_vector(query_embedding, top_k=5)

for score, text_id, vec_id in results:
    print(f"Score: {score:.4f} | {store.read_text(text_id)}")
```

### With OpenAI Embeddings

```python
from contexttape import TSStore, get_client, embed_text_1536
import os

# Set your API key
os.environ["OPENAI_API_KEY"] = "sk-..."

# Initialize
client = get_client()
store = TSStore("my_store")

# Ingest documents
docs = [
    "Python is a versatile programming language.",
    "Neural networks power modern AI systems.",
    "Data preprocessing is crucial for ML success."
]

for doc in docs:
    embedding = embed_text_1536(client, doc)
    store.append_text_with_embedding(doc, embedding, quantize=True)

# Search
query = "artificial intelligence programming"
query_emb = embed_text_1536(client, query)
results = store.search_by_vector(query_emb, top_k=3)

for score, tid, vid in results:
    print(f"{score:.4f}: {store.read_text(tid)}")
```

### High-Level Client API

```python
from contexttape import ContextTapeClient

# Create client (handles embeddings automatically)
client = ContextTapeClient("my_store")

# Ingest documents
client.ingest("Document content here", metadata={"author": "Alice", "date": "2024-01-01"})

# Batch ingest
texts = ["Doc 1", "Doc 2", "Doc 3"]
client.ingest_batch(texts)

# Search
results = client.search("query text", top_k=5)
for result in results:
    print(f"{result.score:.4f}: {result.text}")
    if result.metadata:
        print(f"  Metadata: {result.metadata}")
```

### Command-Line Interface

```bash
# Ingest documents
ct ingest-path ./documents --out-dir my_store --quantize --verbose

# Search
ct search "machine learning" --wiki-dir my_store --topk 5

# Get statistics
ct stat --wiki-dir my_store

# Interactive chat with retrieval
ct chat --wiki-dir my_store --topk 8 --verbose
```

---

## 📚 Examples

Check out the `examples/` directory:

- **[quickstart.py](examples/quickstart.py)** - Basic operations and workflows
- **[advanced_usage.py](examples/advanced_usage.py)** - Advanced patterns and integrations  
- **[tutorial.py](examples/tutorial.py)** - Step-by-step learning tutorials
- **[comprehensive_examples.py](examples/comprehensive_examples.py)** - Production patterns

Run them with:
```bash
python examples/quickstart.py
python examples/tutorial.py
```

---

## 🚀 Ingesting Data

### Option A: wiki corpus
```bash
ct build-wiki \
  --topics-file scripts/topics.example.txt \
  --out-dir wiki_store \
  --verbose
```

### Option B: Local documents
```bash
ct ingest-path ./docs \
  --out-dir wiki_store \
  --exts md txt pdf \
  --max-pdf-pages 10 \
  --verbose
```

During ingestion, each text chunk is embedded and stored as:
```
segment_<n>.is   # text
segment_<n+1>.is # embedding (float32 or int8)
```

Link fields in the headers connect the two.

---

## 🧾 Segment Format

Each `.ts` file contains:
```
[32-byte header][payload]
```

### Header Layout (32 bytes, little-endian)
| Field | Type | Bytes | Description |
|-------|------|--------|-------------|
| next_id | int32 | 4 | link to paired vector/text |
| prev_id | int32 | 4 | reverse link |
| data_len | int32 | 4 | length of payload |
| data_type | int32 | 4 | 0=text, 1=vec_f32, 2=vec_i8, 100=coarse |
| dim | int32 | 4 | vector dimension |
| scale | float32 | 4 | quantization scale |
| reserved | 8 | timestamp / nonce / magic |

---

## 💾 Python API

```python
from contexttape.storage import TSStore, MultiStore, write_playlist
import numpy as np

# Create store
store = TSStore("wiki_store")

# Append text + embedding
vec = np.random.randn(1536).astype(np.float32)
t_id, v_id = store.append_text_with_embedding("Photosynthesis converts light energy.", vec)

# Search
q = np.random.randn(1536).astype(np.float32)
hits = store.search_by_vector(q, top_k=5)
for score, tid, eid in hits:
    print(score, store.read_text(tid))

# Multi-store fusion
wiki = TSStore("wiki_store")
chat = TSStore("chat_ts")
ms = MultiStore([wiki, chat])
res = ms.search(q, per_shard_k=8, final_k=5)
```

---

## 🧩 CLI Reference

### 1️⃣ `ct search`
Search stores for nearest vector matches.

```bash
ct search "photosynthesis basics" \
  --wiki-dir wiki_store \
  --chat-dir chat_ts \
  --topk 5 \
  --verbose
```

### 2️⃣ `ct chat`
Hybrid retrieval + prompt assembly.
```bash
ct chat
```
OR
```bash
ct chat \
  --wiki-dir wiki_store --chat-dir chat_ts \
  --topk 8 \
  --alpha 0.6 \
  --min-score 0.32 --min-lex 0.1 --min-hybrid 0.25 \
  --verbose
```

### 3️⃣ `ct stat`
Show stats.

```bash
ct stat --wiki-dir wiki_store
```

### 4️⃣ `ct reset-chat`
Reset chat history.

```bash
ct reset-chat --chat-dir chat_ts
```

---

## 🧪 Benchmarking

ContextTape includes a microbenchmark that measures:
- Query latency (ms)
- QPS (queries/sec)
- Memory footprint (RSS/PSS)
- Estimated or measured energy (J)
- Corpus size and structure

### Run a benchmark

```bash
# Prepare queries
printf "photosynthesis\nquantum computing\n" > /tmp/queries.txt

# Run benchmark (5 repeats)
ct bench \
  --wiki-dir wiki_store --chat-dir chat_ts \
  --queries-file /tmp/queries.txt \
  --repeats 5 \
  --topk 5 \
  --energy-aware \
  --assume-power-watts 15 \
  --out-json bench/bench.json \
  --out-csv bench/bench.csv \
  --out-md bench/bench.md \
  --verbose
```

| Component              | Description                                        |
| ---------------------- | -------------------------------------------------- |
| **Storage layer**      | File segments with headers and payloads            |
| **Retrieval**          | Sequential scan over vector segments               |
| **Dereference**        | Load text only for top-k                           |
| **Quantization**       | int8 + scale, dequantized on read                  |
| **Hybrid re-rank**     | Vector + lexical similarity                        |
| **Multi-store fusion** | Merge per-shard results                            |
| **Energy module**      | Adjust stride/k under power budget                 |
| **Playlist**           | Optional `.m3u8` listing for streaming replication |


### Sample output (bench/bench.md)

``````text
Latency (ms): p50=47.71, p95=97.59, mean=49.90
Throughput: 18.66 QPS
Memory: RSS=162.9 MB, PSS=153.9 MB
Energy (est): 16.076 J @ 15 W
Segments: 50 pairs, 487k tokens
Corpus: wiki_store=2.50MB, chat_ts=0.00MB





---

✅ **How to run a benchmark (step-by-step)**

1. Make sure you’ve already ingested at least one store (`wiki_store`, `chat_ts`).
2. Create a query file:
   ```bash
   printf "photosynthesis\nquantum computing\n" > /tmp/queries.txt
   ```
3. Run:
   ```bash
   ct bench \
     --wiki-dir wiki_store --chat-dir chat_ts \
     --queries-file /tmp/queries.txt \
     --repeats 5 \
     --topk 5 \
     --energy-aware \
     --assume-power-watts 15 \
     --verbose
   ```
4. Check results:
   - `bench/bench.json` → structured results.
   - `bench/bench.csv` → spreadsheet-ready metrics.
   - `bench/bench.md` → human-readable summary.
   - Energy (if supported) is computed from RAPL, else estimated via `assume-power-watts`.

---

Would you like me to add the **`ct bench` Python implementation section** (the one that measures latency, memory, and energy) into your README as an appendix? That would make the documentation fully reproducible for reviewers or patent enablement.
