# ContextTape: What You Actually Get

## TL;DR

```python
from contexttape import ISStore

store = ISStore("my_data")  # Creates: my_data/
store.append_text_with_embedding("text", embedding)  # Creates: my_data/segment_0.is, segment_1.is
results = store.search_by_vector(query, top_k=5)  # Scans files, returns matches
```

**That's it. No database. Just files.**

---

## File Structure You Get

### After Installation

```
your_project/
├── venv/                         ← Your Python environment
│   └── lib/python3.x/
│       └── site-packages/
│           └── contexttape/      ← Package installed here
│               ├── storage.py
│               ├── embed.py
│               └── ...
└── app.py                        ← Your code
```

### After Running Your Code

```python
# app.py
from contexttape import ISStore

store = ISStore("data/knowledge")
store.append_text("Hello")
```

**Creates:**

```
your_project/
├── app.py
└── data/                         ← YOU create this organization
    └── knowledge/                ← Store directory (auto-created)
        ├── segment_0.is          ← Text: "Hello"
        └── segment_1.is          ← Embedding vector
```

---

## What Gets Created By What

| Your Code | What Gets Created | Location |
|-----------|-------------------|----------|
| `ISStore("mystore")` | Directory: `mystore/` | Current directory |
| `store.append_text("hi")` | File: `mystore/segment_0.is` | In store directory |
| `store.append_vector_i8(vec)` | File: `mystore/segment_1.is` | In store directory |

### Example Session

```python
from contexttape import ISStore
import numpy as np

# This creates: knowledge/
store = ISStore("knowledge")

# This creates: knowledge/segment_0.is (32 bytes header + text)
text_id = store.append_text("Python is great")

# This creates: knowledge/segment_1.is (32 bytes header + 1536 int8 values)
vec = np.random.randn(1536).astype(np.float32)
vec_id = store.append_vector_i8(vec, prev_text_id=text_id)

# This creates: knowledge/segment_2.is and segment_3.is
text_id2 = store.append_text("Machine learning rocks")
vec_id2 = store.append_vector_i8(np.random.randn(1536).astype(np.float32), prev_text_id=text_id2)
```

**Result on disk:**

```bash
$ ls -lh knowledge/
segment_0.is  # 47 bytes  (32 header + 15 text)
segment_1.is  # 1568 bytes (32 header + 1536 int8)
segment_2.is  # 54 bytes  (32 header + 22 text)
segment_3.is  # 1568 bytes (32 header + 1536 int8)
```

---

## Where Should Stores Go?

### ❌ Don't Do This

```python
# Creates stores everywhere!
store1 = ISStore("wiki_store")          # ./wiki_store/
store2 = ISStore("chat_ts")             # ./chat_ts/
store3 = ISStore("embeddings")          # ./embeddings/
# Your project root becomes messy with store directories
```

### ✅ Do This

```python
# Organize under data/
from pathlib import Path

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

wiki = ISStore("data/wiki")
chat = ISStore("data/chat")
embeddings = ISStore("data/embeddings")
```

**Result:**

```
your_project/
├── app.py
├── data/                  ← All stores here
│   ├── wiki/
│   │   └── segment_*.is
│   ├── chat/
│   │   └── segment_*.is
│   └── embeddings/
│       └── segment_*.is
└── .gitignore             ← Add: data/
```

---

## Package vs User Data

### What's in the Package (site-packages/contexttape/)

```
contexttape/                   ← Installed library
├── __init__.py
├── storage.py                 ← ISStore class
├── embed.py                   ← Embedding functions
├── integrations.py            ← ContextTapeClient
├── search.py                  ← Search algorithms
├── cli.py                     ← CLI commands
└── ...
```

**This is source code. You import it.**

### What's in Your Project (user data)

```
your_app/
├── app.py                     ← Your code
└── data/                      ← Your data stores
    └── my_store/
        └── segment_*.is       ← Your segment files
```

**This is user data. Your app creates it.**

---

## Testing What Gets Created

### Run the Test Script

```bash
cd contexttape
./run_test.sh
```

**Output shows:**

```
📁 Files created in test_data/basic_store:
   Total: 6 segment files
   - segment_0.is (64 bytes)
   - segment_1.is (1,568 bytes)
   - segment_2.is (64 bytes)
   ...

📂 All test data is in: /home/user/contexttape/test_data/

Stores created:
  📁 basic_store/ (6 segments, 4.8 KB)
  📁 search_store/ (6 segments, 4.7 KB)
  📁 client_store/ (6 segments, 18.3 KB)
  ...
```

### Clean Up Test Data

```bash
./run_test.sh --cleanup
```

---

## Integration Examples

### Minimal App

```python
# app.py
from contexttape import ISStore
import numpy as np

def get_embedding(text):
    # Your embedding function (OpenAI, local model, etc.)
    return np.random.randn(1536).astype(np.float32)

# Create store
store = ISStore("data/rag_store")

# Add documents
docs = ["Doc 1", "Doc 2", "Doc 3"]
for doc in docs:
    text_id = store.append_text(doc)
    vec_id = store.append_vector_i8(get_embedding(doc), prev_text_id=text_id)

# Search
query_vec = get_embedding("search query")
results = store.search_by_vector(query_vec, top_k=5)

for score, text_id, vec_id in results:
    print(f"{score:.3f}: {store.read_text(text_id)}")
```

**Run it:**

```bash
python app.py
```

**Creates:**

```
data/
└── rag_store/
    ├── segment_0.is  # "Doc 1" text
    ├── segment_1.is  # Doc 1 embedding
    ├── segment_2.is  # "Doc 2" text
    ├── segment_3.is  # Doc 2 embedding
    ├── segment_4.is  # "Doc 3" text
    └── segment_5.is  # Doc 3 embedding
```

### With Proper Organization

```python
# config.py
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

STORES = {
    "knowledge": DATA_DIR / "knowledge",
    "chat": DATA_DIR / "chat",
    "cache": DATA_DIR / "cache",
}

# app.py
from contexttape import ISStore
from config import STORES

knowledge_store = ISStore(str(STORES["knowledge"]))
chat_store = ISStore(str(STORES["chat"]))
cache_store = ISStore(str(STORES["cache"]))
```

---

## What About runner.py?

**Q: What is `/home/doom/RAGLite/runner.py`?**

**A:** Old experimental code. **Ignore it.** It's not part of the package.

**Use instead:**
- `test_system.py` — Test the package works
- `run_test.sh` — Shell script to run tests
- `examples/` — Usage examples
- `ct` CLI — After `pip install contexttape`

---

## Quick Reference

### Create Store

```python
from contexttape import ISStore
store = ISStore("path/to/store")  # Creates directory
```

### Add Data

```python
# Text only
text_id = store.append_text("text")

# Text + Vector (linked)
text_id = store.append_text("text")
vec_id = store.append_vector_i8(embedding, prev_text_id=text_id)

# Or combined
text_id, vec_id = store.append_text_with_embedding("text", embedding)
```

### Search

```python
results = store.search_by_vector(query_embedding, top_k=5)
for score, text_id, vec_id in results:
    text = store.read_text(text_id)
    print(f"{score}: {text}")
```

### Multiple Stores

```python
from contexttape import MultiStore

wiki = ISStore("data/wiki")
docs = ISStore("data/docs")

multi = MultiStore([wiki, docs])
results = multi.search(query_vec, final_k=10)
```

---

## Summary

| Question | Answer |
|----------|--------|
| **What gets installed?** | Python package in site-packages/contexttape/ |
| **What gets created when I use it?** | Directories with segment_*.is files |
| **Where should stores go?** | Under data/ in your project |
| **How do I test it?** | Run `./run_test.sh` |
| **What's runner.py?** | Ignore it (legacy code) |
| **Where's my data?** | In the directories you specify: `ISStore("path")` |
| **Do I need a database?** | No |
| **Do I need a server?** | No |
| **What if I want to move data?** | Just copy/move the store directory |

---

**Bottom line:** ContextTape creates directories with `.ts` files. That's your database. Copy it, move it, version it, ship it. It's just files.
