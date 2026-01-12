\
# Python API Usage

Below are minimal end-to-end examples. You can build stores entirely in code.

## Create a store & add text

```python
from contexttape.storage import TSStore
from contexttape.embed import get_client, embed_text_1536

store = TSStore("wiki_store")
client = get_client()

text = "Neural networks are function approximators composed of layers."
vec = embed_text_1536(client, text)  # dynamic token-aware chunking
tid, vid = store.append_text_with_embedding(text, vec)
```

## Ingest a directory (programmatic)

```python
from contexttape.storage import TSStore
from contexttape.embed import get_client, embed_text_1536
from contexttape.ingest_generic import iter_files

store = TSStore("wiki_store")
client = get_client()

for path, text in iter_files("./docs", exts=["md", "txt"]):
    emb = embed_text_1536(client, text)
    store.append_text_with_embedding(text, emb)
```

## Hybrid search

```python
from contexttape.search import combined_search
from contexttape.embed import embed_text_1536, get_client
from contexttape.storage import TSStore

store = TSStore("wiki_store")
client = get_client()

q = "how does backpropagation work"
qvec = embed_text_1536(client, q)
hits = combined_search(q, qvec, wiki_store=store, chat_store=store, top_k=5, alpha=0.6)
for src, score, tid, vid in hits:
    print(src, f"{score:.4f}", tid, vid, store.read_text(tid)[:120])
```

## Chat with retrieved context (high-level)

Use the CLI (`ct chat`) for the full pipeline. For custom apps, see `cmd_chat` in `contexttape/cli.py`.\n