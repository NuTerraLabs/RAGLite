\
# Quickstart

## 1) Install & set key

```bash\npip install contexttape  # or: pip install -e .\n```

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
ct chat --wiki-dir wiki_store --chat-dir chat_ts --topk 8 --alpha 0.6         --min-lex 0.12 --min-hybrid 0.28 --max-context-blocks 5 --verbose
```

## 5) Benchmark (optional)

```bash
ct bench --wiki-dir wiki_store --chat-dir chat_ts --repeats 3 --topk 8 --verbose          --out-json bench.json --out-csv bench.csv --out-md bench.md
```\n