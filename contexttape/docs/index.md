# ContextTape\n\nSee the README at project root, or keep reading.\n\n\
# ContextTape

**Lightweight local RAG: build simple on-disk text/vector stores, search them, and chat with hybrid retrieval.**

- **Zero infra**: for text + embeddings  
- **Flexible ingestion**: pages, paths, PDFs, images/audio (placeholders), text, documents, mixed corpora  
- **Hybrid search**: vector + lexical blend with tunable α and thresholds  
- **Nice CLI** *and* clean Python API



## Install

```bash\npip install contexttape  # or: pip install -e .\n```

## Quick Start

```bash\npython scripts/seed_multimodal_corpus.py --out sample_corpus\nct ingest-any sample_corpus --out-dir wiki_store --quantize --verbose\n\n# chat\nct chat --wiki-dir wiki_store --chat-dir chat_ts --topk 8 --alpha 0.6 --verbose\n```

## Use in Python

```python\nfrom contexttape.storage import TSStore\nfrom contexttape.embed import get_client, embed_text_1536\nfrom contexttape.search import combined_search\n\nstore = TSStore('wiki_store')\nclient = get_client()\n\ntext = 'Neural networks are function approximators composed of layers.'\nvec = embed_text_1536(client, text)\ntid, vid = store.append_text_with_embedding(text, vec)\n\nq = 'how does backpropagation work'\nqvec = embed_text_1536(client, q)\nhits = combined_search(q, qvec, wiki_store=store, chat_store=store, top_k=5, alpha=0.6)\nfor src, score, tid, vid in hits:\n    print(src, f'{score:.4f}', tid, vid, store.read_text(tid)[:120])\n```

## CLI Overview

See the [CLI Reference](docs/cli.md) for all commands and examples, or run:

```bash\nct --help\nct <subcommand> --help\n```

## Configuration (env)

- `OPENAI_API_KEY` – required for embeddings/chat (e.g., `sk-...`)
- `WIKI_TS_DIR` – default wiki store directory (e.g., `wiki_store`)
- `CHAT_TS_DIR` – default chat store directory (e.g., `chat_ts`)
- `TOP_K` – default `topk`
- `DEBUG_DIR` – verbose debug dump directory
- `CTX_SEG_EXT` – custom segment file extension (e.g. `.ismail`)
- Embedding behavior (dynamic chunking):
  - `EMBED_MODEL_MAX_TOKENS` (default `8192`)
  - `EMBED_CHUNK_TOKENS` (default `800`)
  - `EMBED_CHUNK_OVERLAP` (default `200`)

## Docs Site

We ship an MkDocs site (Material theme). Build locally:

```bash\npip install -r docs/requirements-docs.txt\nmkdocs serve\n```

Deploy automatically via GitHub Pages using the provided workflow.\n