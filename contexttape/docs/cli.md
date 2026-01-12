# CLI Reference\n\nThis page is auto-generated from the CLI `--help` output.\n\n## Top-level\n\n```text\nusage: ct [-h] [--version]
          {build-wiki,build-wikipedia,ingest-path,search,chat,stat,bench,reset-chat,ingest-any}
          ...

ContextTape CLI – build simple on-disk text/vector segment stores and run hybrid RAG.
Use the subcommands below. Each subcommand has examples and templates.

positional arguments:
  {build-wiki,build-wikipedia,ingest-path,search,chat,stat,bench,reset-chat,ingest-any}
    build-wiki (build-wikipedia)
                        Ingest Wikipedia pages by title
    ingest-path         Ingest all files under a path
    search              Search across wiki and chat memory (env: WIKI_TS_DIR,
                        CHAT_TS_DIR)
    chat                Interactive chat using retrieved context (env:
                        WIKI_TS_DIR, CHAT_TS_DIR)
    stat                Show store stats (env: WIKI_TS_DIR, CHAT_TS_DIR)
    bench               Benchmark: tokens, size, speed, RAM, energy
    reset-chat          Clear chat store (env: CHAT_TS_DIR)
    ingest-any          Ingest arbitrary files
                        (text/image/audio/pdf/video/other blobs)

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit

Environment variables
---------------------
  OPENAI_API_KEY   Required for embeddings/chat (e.g., "sk-...").
  WIKI_TS_DIR      Default wiki store directory (default: wiki_real_ts).
  CHAT_TS_DIR      Default chat store directory (default: chat_ts).
  TOP_K            Default top-k (default: 5).
  DEBUG_DIR        Where --verbose logs go (default: debug).
  CTX_SEG_EXT      Optional custom segment file extension (e.g. ".ismail").

Quick start
-----------
  # 1) Install (venv recommended)
  pip install -e .

  # 2) Set API key for embeddings/chat
  export OPENAI_API_KEY="sk-..."

  # 3) Seed a small multimodal corpus then ingest
  python scripts/seed_multimodal_corpus.py --out sample_corpus
  ct ingest-any sample_corpus --out-dir wiki_store --quantize --verbose

  # 4) Chat with retrieval
  ct chat --wiki-dir wiki_store --chat-dir chat_ts --topk 8 --alpha 0.55           --min-lex 0.15 --min-hybrid 0.28 --max-context-blocks 5 --verbose

Tip: Run "ct <command> --help" for command-specific examples and templates.\n```\n\n\n