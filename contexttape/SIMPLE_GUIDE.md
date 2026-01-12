# ContextTape: Simple Usage Guide

## What Actually Happens When You Use This

### 1. **You Create a Store = A Directory Gets Created**

```python
from contexttape import ISStore

store = ISStore("my_data")  # Creates directory: my_data/
```

**Result:** `my_data/` directory is created where you run your code.

### 2. **You Add Data = Segment Files Get Created**

```python
# Add text + embedding
text_id = store.append_text("Hello world")
vec_id = store.append_vector_i8(embedding, prev_text_id=text_id)
```

**Result:** Two files created:
- `my_data/segment_0.is` (text: "Hello world")
- `my_data/segment_1.is` (embedding vector)

### 3. **You Search = Files Get Scanned**

```python
results = store.search_by_vector(query_embedding, top_k=5)
```

**What happens:**
- Scans `segment_*.is` files for vector matches
- Returns top 5 most similar
- NO database, NO server, just file I/O

---

## What Gets Created

```
your_project/
├── app.py                    ← Your code
└── my_data/                  ← Store directory (auto-created)
    ├── segment_0.is          ← Text segment
    ├── segment_1.is          ← Vector segment
    ├── segment_2.is          ← Text segment
    ├── segment_3.is          ← Vector segment
    └── ...
```

**That's it. No databases, no config files, just `.ts` segment files.**

---

## Recommended Structure for Your App

```
your_app/
├── src/
│   └── app.py                ← Your application code
├── data/                     ← All ContextTape stores here
│   ├── knowledge/            ← Knowledge base store
│   │   ├── segment_0.is
│   │   ├── segment_1.is
│   │   └── ...
│   ├── chat_history/         ← Chat logs store
│   │   └── ...
│   └── embeddings/           ← Pre-computed embeddings
│       └── ...
└── .gitignore                ← Add: data/
```

**In your code:**

```python
from contexttape import ISStore

# All stores under data/
knowledge = ISStore("data/knowledge")
chat = ISStore("data/chat_history")
embeddings = ISStore("data/embeddings")
```

---

## Quick Test Script

**File:** `test_system.py`

```bash
# Run the included test
cd contexttape
PYTHONPATH=src:$PYTHONPATH python test_system.py

# See what gets created
python test_system.py --only-show

# Clean up after
python test_system.py --cleanup
```

**What it shows:**
1. Creates `test_data/` directory
2. Creates multiple stores (basic_store, search_store, etc.)
3. Shows you what files get created
4. Demonstrates search
5. Cleans up when done

---

## Integration in Your Project

### Option 1: As a Library (Recommended)

```python
# your_app/app.py
from contexttape import ISStore, ContextTapeClient
import numpy as np

# Simple API
store = ISStore("data/my_store")
store.append_text_with_embedding("text", embedding)

# Or high-level client
client = ContextTapeClient("data/my_store", embed_fn=my_embed_func)
client.ingest("Document text")
results = client.search("query", top_k=5)
```

### Option 2: Via CLI

```bash
# Install package
pip install contexttape

# Use CLI
ct ingest-path ./docs --out-dir data/knowledge
ct search "query text" --topk 5
ct chat --topk 3
```

---

## What's in src/contexttape/?

**Core files you'll use:**

| File | Purpose | You Use This When... |
|------|---------|---------------------|
| **storage.py** | Core storage engine | Creating stores, adding data, searching |
| **embed.py** | Embedding generation | Need OpenAI embeddings |
| **integrations.py** | High-level client | Want easy API (ingest, search) |
| **search.py** | Search algorithms | Need custom search logic |

**Supporting files:**

| File | Purpose |
|------|---------|
| cli.py | Command-line interface |
| ingest_*.py | Document ingestion |
| energy.py | Energy monitoring (optional) |
| config.py | Configuration |

**You mostly use:** `ISStore` from `storage.py` OR `ContextTapeClient` from `integrations.py`

---

## Common Patterns

### Pattern 1: Simple RAG

```python
from contexttape import ISStore
import numpy as np

# Create store
store = ISStore("data/rag_store")

# Ingest documents
for doc in documents:
    embedding = get_embedding(doc)  # Your embedding function
    text_id = store.append_text(doc)
    vec_id = store.append_vector_i8(embedding, prev_text_id=text_id)

# Search
query_embedding = get_embedding(user_query)
results = store.search_by_vector(query_embedding, top_k=5)

# Use results
for score, text_id, vec_id in results:
    context = store.read_text(text_id)
    # Send to LLM...
```

### Pattern 2: Multiple Stores

```python
from contexttape import ISStore, MultiStore

# Create separate stores
wiki = ISStore("data/wikipedia")
docs = ISStore("data/internal_docs")
chat = ISStore("data/chat_logs")

# Search across all
multi = MultiStore([wiki, docs, chat])
results = multi.search(query_vec, final_k=10)
```

### Pattern 3: High-Level Client

```python
from contexttape import ContextTapeClient

def my_embedder(text):
    # Your embedding logic
    return np.array([...])

client = ContextTapeClient("data/store", embed_fn=my_embedder)

# Ingest
client.ingest("Document text")

# Search
results = client.search("query", top_k=5)
for result in results:
    print(f"{result.score}: {result.text}")
```

---

## File Format

Each `.ts` file is:
```
[32-byte header][encrypted payload]
```

**Header contains:**
- Next segment ID
- Previous segment ID (linking text ↔ vector)
- Data length
- Data type (text, vector, etc.)
- Embedding dimension
- Quantization scale

**Payload is:**
- UTF-8 text for text segments
- Float32 or Int8 arrays for vector segments

---

## What About runner.py?

`runner.py` in the repo root is **legacy/experimental code**. Ignore it.

**Use instead:**
- `test_system.py` — Test the package
- Examples in `examples/` — Learn usage patterns
- CLI commands — `ct` command after install

---

## Quick Start Checklist

1. **Install:**
   ```bash
   pip install contexttape
   # OR for development:
   cd contexttape && pip install -e .
   ```

2. **Create a store:**
   ```python
   from contexttape import ISStore
   store = ISStore("data/my_store")
   ```

3. **Add data:**
   ```python
   text_id = store.append_text("text")
   vec_id = store.append_vector_i8(embedding, prev_text_id=text_id)
   ```

4. **Search:**
   ```python
   results = store.search_by_vector(query_vec, top_k=5)
   ```

5. **Your data lives in:** `data/my_store/segment_*.is`

---

## Key Takeaways

✅ **ContextTape creates directories with `.ts` segment files**  
✅ **Each store is a directory, nothing more**  
✅ **No database, no server, no infrastructure**  
✅ **Organize stores under `data/` in your project**  
✅ **Use `ISStore` for basic ops or `ContextTapeClient` for high-level API**  
✅ **Test with `test_system.py` to see what gets created**  

---

## Next Steps

1. **Run the test:** `python test_system.py`
2. **Read examples:** Check `examples/quickstart.py`
3. **Try in your project:** Create a store and add data
4. **Organize data:** Put stores under `data/` directory
5. **Integrate:** Use in your RAG pipeline

**Questions?** Read the source:
- [storage.py](src/contexttape/storage.py) — Core logic (617 lines)
- [integrations.py](src/contexttape/integrations.py) — High-level API

---

**That's it. No magic, no complexity. Just files.**
