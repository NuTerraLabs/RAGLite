# ContextTape Quick Reference

## Installation
```bash
pip install contexttape
```

## Basic Patterns

### 1. Create and Use a Store
```python
from contexttape import ISStore
import numpy as np

store = ISStore("my_store")
text_id = store.append_text("Hello World")
retrieved = store.read_text(text_id)
```

### 2. Store with Embeddings
```python
text = "Document text"
embedding = np.random.randn(1536).astype(np.float32)
text_id, vec_id = store.append_text_with_embedding(text, embedding, quantize=True)
```

### 3. Search
```python
query_vec = np.random.randn(1536).astype(np.float32)
results = store.search_by_vector(query_vec, top_k=5)

for score, text_id, vec_id in results:
    print(f"{score:.4f}: {store.read_text(text_id)}")
```

### 4. OpenAI Embeddings
```python
from contexttape import get_client, embed_text_1536
import os

os.environ["OPENAI_API_KEY"] = "sk-..."
client = get_client()

embedding = embed_text_1536(client, "Your text here")
```

### 5. High-Level Client
```python
from contexttape import ContextTapeClient

client = ContextTapeClient("my_store")
client.ingest("Document content")
results = client.search("query", top_k=5)
```

### 6. Multi-Store Search
```python
from contexttape import ISStore, MultiStore

store1 = ISStore("store1")
store2 = ISStore("store2")
multi = MultiStore([store1, store2])

results = multi.search(query_vec, per_shard_k=5, final_k=10)
```

### 7. Batch Operations
```python
texts = ["Doc 1", "Doc 2", "Doc 3"]
embeddings = [embed_text_1536(client, t) for t in texts]
results = store.append_batch(texts, embeddings, quantize=True)
```

### 8. Store Statistics
```python
stats = store.stat()
print(f"Pairs: {stats['pairs']}")
print(f"Next ID: {stats['next_id']}")
```

### 9. Export Data
```python
export = store.export_to_dict(include_vectors=False)
import json
with open("backup.json", "w") as f:
    json.dump(export, f)
```

### 10. Store Maintenance
```python
# Compact store (remove orphaned segments)
stats = store.compact()
print(f"Deleted {stats['deleted_segments']} orphaned segments")

# Delete specific segment
store.delete_segment(segment_id)
```

## CLI Commands

```bash
# Ingest documents from directory
ct ingest-path ./docs --out-dir my_store --quantize --verbose

# Search
ct search "machine learning" --wiki-dir my_store --topk 5

# Interactive chat with retrieval
ct chat --wiki-dir my_store --topk 8

# Statistics
ct stat --wiki-dir my_store

# Benchmark
ct bench --wiki-dir my_store --queries-file queries.txt --repeats 5
```

## Common Patterns

### RAG Pipeline
```python
from contexttape import ISStore, get_client, embed_text_1536

# Setup
client = get_client()
store = ISStore("knowledge_base")

# Ingest
documents = ["Doc 1...", "Doc 2...", "Doc 3..."]
for doc in documents:
    emb = embed_text_1536(client, doc)
    store.append_text_with_embedding(doc, emb, quantize=True)

# Query
query = "What is machine learning?"
query_emb = embed_text_1536(client, query)
results = store.search_by_vector(query_emb, top_k=3)

# Build context
context = "\n\n".join(store.read_text(tid) for _, tid, _ in results)
# Send to LLM with context...
```

### With Metadata
```python
import json

# Store with metadata
metadata = {"author": "Alice", "date": "2024-01-01"}
text_with_meta = json.dumps({"text": "Content", "meta": metadata})
emb = embed_text_1536(client, "Content")
tid, vid = store.append_text_with_embedding(text_with_meta, emb)

# Retrieve with metadata
results = store.search_by_vector(query_emb, top_k=1)
for _, tid, _ in results:
    data = json.loads(store.read_text(tid))
    print(data["text"])
    print(data["meta"])
```

### Filtered Search
```python
# Store with category tags
docs_with_tags = [
    ("Doc about AI", "tech"),
    ("Doc about cooking", "food"),
    ("Doc about ML", "tech"),
]

doc_metadata = []
for text, category in docs_with_tags:
    emb = embed_text_1536(client, text)
    tid, vid = store.append_text_with_embedding(text, emb)
    doc_metadata.append((tid, category))

# Search with filter
results = store.search_by_vector(query_emb, top_k=10)
tech_results = [
    (score, tid, vid) 
    for score, tid, vid in results 
    if any(t == tid and cat == "tech" for t, cat in doc_metadata)
]
```

## Performance Tips

1. **Use quantization** for 4x space savings: `quantize=True`
2. **Batch operations** for efficiency: `append_batch()`
3. **Use stride** for faster approximate search: `stride=2`
4. **Regular compaction** to remove orphaned segments
5. **Multi-store** for different data types or hot/cold separation

## Data Types

- `DT_TEXT` (0): UTF-8 text
- `DT_VEC_F32` (1): Float32 vectors
- `DT_VEC_I8` (2): Int8 quantized vectors
- `DT_JSON` (13): JSON metadata
- `DT_BLOB` (12): Binary data

## Links

- **Documentation**: [docs/](docs/)
- **Examples**: [examples/](examples/)
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)
- **Tests**: `pytest tests/ -v`
- **Verification**: `python verify_setup.py`
