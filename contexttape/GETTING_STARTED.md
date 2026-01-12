# Getting Started with ContextTape

## What is ContextTape?

**A file-based RAG (Retrieval-Augmented Generation) system.** No vector database needed.

Perfect for:
- ✅ Building RAG applications with LLMs (ChatGPT, Claude, etc.)
- ✅ Semantic search over your documents
- ✅ Question-answering systems with memory
- ✅ Portable AI that works offline

## Install (10 seconds)

```bash
pip install contexttape
export OPENAI_API_KEY="sk-..."
```

## Build Your First RAG (60 seconds)

### 1. Create Topics File

```bash
echo "Python_(programming_language)
Machine_learning
Artificial_intelligence" > topics.txt
```

### 2. Ingest Wikipedia Knowledge

```bash
ct build-wiki --topics-file topics.txt --limit 3 --verbose
```

This creates `data/wiki/segment_*.is` files.

### 3. Search It

```bash
ct search "What is machine learning?"
```

Output:
```
[TOP 1] score=0.64 text_seg=2 emb_seg=3
  Machine learning (ML) is a field of study in artificial intelligence...
```

### 4. Chat With It

```bash
ct chat
```

Interactive Q&A using your knowledge base!

## Python API (Simple)

```python
from contexttape import ISStore
import numpy as np

# Create a store
store = ISStore("data/my_knowledge")

# Add documents with embeddings
from contexttape import get_client, embed_text_1536

client = get_client()  # Uses OPENAI_API_KEY
text = "Python is a programming language"
embedding = embed_text_1536(client, text)

text_id, vec_id = store.append_text_with_embedding(text, embedding)

# Search
query = "What is Python?"
query_embedding = embed_text_1536(client, query)
results = store.search_by_vector(query_embedding, top_k=5)

for score, text_id, vec_id in results:
    text = store.read_text(text_id)
    print(f"Score: {score:.2f} - {text[:100]}...")
```

## High-Level API (Even Simpler)

```python
from contexttape import ContextTapeClient

# Initialize
client = ContextTapeClient("data/my_rag")

# Ingest documents (auto-embeds with OpenAI)
client.ingest("Machine learning is a subset of AI...")
client.ingest("Python is widely used in data science...")

# Search
results = client.search("Tell me about Python", top_k=3)

for result in results:
    print(f"{result.score:.2f}: {result.text[:100]}...")
```

## Where Does Data Go?

```
your_project/
├── data/
│   ├── wiki/              ← Wikipedia knowledge (ct build-wiki)
│   │   ├── segment_0.is   ← Text
│   │   ├── segment_1.is   ← Vector embeddings
│   │   ├── segment_2.is
│   │   └── ...
│   └── chat/              ← Chat history (ct chat)
│       ├── segment_0.is
│       └── ...
└── your_code.py
```

**Your data stays in your project.** Package code is in `site-packages/contexttape/`.

## Key Features

| Feature | ContextTape | Traditional Vector DBs |
|---------|-------------|------------------------|
| **Setup** | `pip install contexttape` | Docker, servers, cloud accounts |
| **Storage** | `.is` files | Database servers |
| **Portability** | `cp -r data/` | Database dumps, migrations |
| **Memory** | ~150MB for 500K tokens | GBs of indexes |
| **Cold Start** | Instant (mmap) | Index loading time |
| **Version Control** | Git-friendly files | Not practical |

## Common Patterns

### 1. Wikipedia RAG

```bash
ct build-wiki --topics-file topics.txt
ct chat
```

### 2. Document RAG

```bash
ct ingest-path ./docs --out-dir data/docs
ct search "your query" --wiki-dir data/docs
```

### 3. Multi-Source RAG

```python
from contexttape import ISStore, MultiStore

wiki = ISStore("data/wiki")
docs = ISStore("data/docs")
chat = ISStore("data/chat")

# Search across all
multi = MultiStore([wiki, docs, chat])
results = multi.search(query_vec, per_shard_k=5, final_k=10)
```

## Next Steps

- **[Full Documentation](README.md)** - Complete API reference
- **[Examples](examples/)** - Real-world use cases
- **[Integrations](README.md#integrations)** - LangChain, LlamaIndex, FastAPI
- **[Benchmarks](README.md#benchmarks)** - Performance comparisons

## Need Help?

- 📖 Read [SIMPLE_GUIDE.md](SIMPLE_GUIDE.md) for common patterns
- 🔍 Check [QUICKSTART.md](QUICKSTART.md) for 60-second demos
- 💬 Open an issue on GitHub
- 📧 Email: info@nuterralabs.com

## Why ContextTape?

Because RAG should be **simple**:
- No database setup
- No cloud dependencies  
- No complex configurations
- Just files and Python

Start building AI apps in 60 seconds instead of 60 minutes.
