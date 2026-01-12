#!/usr/bin/env python3
"""
Generate docs for ContextTape:
- README.md (concise, polished)
- docs/ site (MkDocs Material): index, quickstart, cli reference (auto from --help), python API usage
- .github/workflows/docs.yml (GitHub Pages)
- mkdocs.yml + docs/requirements-docs.txt

Run from repo root:
  python scripts/gen_docs.py
"""

from __future__ import annotations
import re
import shlex
import sys
import subprocess
from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
GHA = REPO / ".github" / "workflows"
DOCS.mkdir(parents=True, exist_ok=True)
GHA.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def run_cmd(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except subprocess.CalledProcessError as e:
        return e.output.strip()

def cli_candidates() -> list[list[str]]:
    # Prefer installed entrypoint, but fall back to module
    return [["ct", "--help"], [sys.executable, "-m", "contexttape.cli", "--help"]]

def help_text(cmd: list[str]) -> str | None:
    out = run_cmd(cmd)
    if any(s in out for s in ("usage:", "ContextTape CLI", "positional arguments:")):
        return out
    return None

def get_ct_help() -> tuple[tuple[str, str], list[str]]:
    help_out = None
    used: list[str] | None = None
    for c in cli_candidates():
        h = help_text(c)
        if h:
            help_out = h
            used = c[:-1]  # base command without --help
            break
    if not help_out or not used:
        print("ERROR: Could not run `ct --help` or `python -m contexttape.cli --help`", file=sys.stderr)
        sys.exit(1)

    # Extract subcommand names from help (robust-ish to argparse variants)
    subcmds: list[str] = []
    for m in re.finditer(r"\\n\\s{2,}([a-z0-9][a-z0-9\\-]+)\\s", help_out):
        name = m.group(1).strip()
        if name and name not in subcmds:
            subcmds.append(name)
    subcmds = sorted(set(subcmds))
    return (" ".join(used), help_out), subcmds

def get_subcommand_help(base_cmd: str, sub: str) -> str:
    parts = shlex.split(base_cmd)
    return run_cmd(parts + [sub, "--help"])

def codeblock(lang: str, text: str) -> str:
    return f"```{lang}\\n{(text or '').strip()}\\n```"

def write_text(path: Path, s: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s.rstrip() + "\\n", encoding="utf-8")

# --------------------------------------------------------------------------------------
# Content builders
# --------------------------------------------------------------------------------------
README_TMPL = """\\
# ContextTape

**Lightweight local RAG: build simple on-disk text/vector stores, search them, and chat with hybrid retrieval.**

- **Zero infra**: local folders for text + embeddings  
- **Flexible ingestion**: wiki pages, paths, PDFs, images/audio (placeholders), mixed corpora  
- **Hybrid search**: vector + lexical blend with tunable α and thresholds  
- **Nice CLI** *and* clean Python API

{badges}

## Install

{code_install}

## Quick Start

{code_quickstart}

## Use in Python

{code_python_api}

## CLI Overview

See the [CLI Reference](docs/cli.md) for all commands and examples, or run:

{code_help}

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

{code_docs_local}

Deploy automatically via GitHub Pages using the provided workflow.
"""

QUICKSTART_MD = """\\
# Quickstart

## 1) Install & set key

{code_install}

```bash
export OPENAI_API_KEY="sk-..."  # required
```

## 2) Seed a small corpus & ingest

```bash
python scripts/seed_multimodal_corpus.py --out sample_corpus
ct ingest-any sample_corpus --out-dir wiki_store --quantize --verbose
```

## 3) Search

```bash
ct search "quantum entanglement applications" --topk 8 --verbose
```

## 4) Chat (hybrid RAG)

```bash
ct chat --wiki-dir wiki_store --chat-dir chat_ts --topk 8 --alpha 0.6 \
        --min-lex 0.12 --min-hybrid 0.28 --max-context-blocks 5 --verbose
```

## 5) Benchmark (optional)

```bash
ct bench --wiki-dir wiki_store --chat-dir chat_ts --repeats 3 --topk 8 --verbose \
         --out-json bench.json --out-csv bench.csv --out-md bench.md
```
"""

PY_API_MD = """\\
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

Use the CLI (`ct chat`) for the full pipeline. For custom apps, see `cmd_chat` in `contexttape/cli.py`.
"""

MKDOCS_YML = """\\
site_name: ContextTape
site_description: Lightweight local RAG: build on-disk text/vector stores, hybrid search, chat.
theme:
  name: material
  features:
    - navigation.sections
    - navigation.expand
    - content.code.copy
repo_url: https://github.com/YOUR_ORG/YOUR_REPO
nav:
  - Home: index.md
  - Quickstart: quickstart.md
  - CLI Reference: cli.md
  - Python API: python_api.md
markdown_extensions:
  - admonition
  - toc:
      permalink: true
"""

DOCS_REQS = """\\
mkdocs>=1.6.0
mkdocs-material>=9.5.0
"""

GHA_WORKFLOW = """\\
name: docs
on:
  push:
    branches: [ main, master ]
  workflow_dispatch: {}
jobs:
  build-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r docs/requirements-docs.txt
      - run: mkdocs build --strict
      - uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ '{{' }} secrets.GITHUB_TOKEN {{ '}}' }}
          publish_dir: ./site
"""

# --------------------------------------------------------------------------------------
# Main generation
# --------------------------------------------------------------------------------------
def main() -> None:
    (base_cmd, root_help), subcommands = get_ct_help()

    # Pull per-subcommand help
    sub_sections = []
    for sub in subcommands:
        sub_help = get_subcommand_help(base_cmd, sub)
        if not sub_help or "usage:" not in sub_help:
            continue
        section = f"## `{sub}`\\n\\n" + codeblock("text", sub_help)
        sub_sections.append(section)

    cli_md = (
        "# CLI Reference\\n\\n"
        "This page is auto-generated from the CLI `--help` output.\\n\\n"
        "## Top-level\\n\\n" + codeblock("text", root_help) + "\\n\\n" +
        "\\n\\n".join(sub_sections)
    )

    # Compose README
    badges = ""
    code_install = codeblock("bash", "pip install contexttape  # or: pip install -e .")
    code_help = codeblock("bash", f"{base_cmd} --help\\n{base_cmd} <subcommand> --help")
    code_quickstart = codeblock("bash", "\\n".join([
        "python scripts/seed_multimodal_corpus.py --out sample_corpus",
        "ct ingest-any sample_corpus --out-dir wiki_store --quantize --verbose",
        "",
        "# chat",
        "ct chat --wiki-dir wiki_store --chat-dir chat_ts --topk 8 --alpha 0.6 --verbose",
    ]))
    code_docs_local = codeblock("bash", "\\n".join([
        "pip install -r docs/requirements-docs.txt",
        "mkdocs serve",
    ]))

    readme = README_TMPL.format(
        badges=badges,
        code_install=code_install,
        code_quickstart=code_quickstart,
        code_python_api=codeblock("python", "\\n".join([
            "from contexttape.storage import TSStore",
            "from contexttape.embed import get_client, embed_text_1536",
            "from contexttape.search import combined_search",
            "",
            "store = TSStore('wiki_store')",
            "client = get_client()",
            "",
            "text = 'Neural networks are function approximators composed of layers.'",
            "vec = embed_text_1536(client, text)",
            "tid, vid = store.append_text_with_embedding(text, vec)",
            "",
            "q = 'how does backpropagation work'",
            "qvec = embed_text_1536(client, q)",
            "hits = combined_search(q, qvec, wiki_store=store, chat_store=store, top_k=5, alpha=0.6)",
            "for src, score, tid, vid in hits:",
            "    print(src, f'{score:.4f}', tid, vid, store.read_text(tid)[:120])",
        ])),
        code_help=code_help,
        code_docs_local=code_docs_local,
    )

    # Write files
    write_text(REPO / "README.md", readme)
    write_text(DOCS / "index.md", "# ContextTape\\n\\nSee the README at project root, or keep reading.\\n\\n" + readme)
    write_text(DOCS / "quickstart.md", QUICKSTART_MD.format(code_install=codeblock("bash", "pip install contexttape  # or: pip install -e .")))
    write_text(DOCS / "cli.md", cli_md)
    write_text(DOCS / "python_api.md", PY_API_MD)
    write_text(REPO / "mkdocs.yml", MKDOCS_YML)
    write_text(DOCS / "requirements-docs.txt", DOCS_REQS)
    write_text(GHA / "docs.yml", GHA_WORKFLOW)

    print("✅ Docs generated:")
    print(" - README.md")
    print(" - docs/index.md, quickstart.md, cli.md, python_api.md")
    print(" - mkdocs.yml")
    print(" - docs/requirements-docs.txt")
    print(" - .github/workflows/docs.yml")
    print("\\nNext:")
    print("  1) pip install -r docs/requirements-docs.txt")
    print("  2) mkdocs serve  # local preview at http://127.0.0.1:8000")
    print("  3) Push to main; GitHub Pages workflow will publish automatically.")

if __name__ == "__main__":
    main()
