# ContextTape - 60 Second Quick Start

## Install

```bash
pip install -e .
export OPENAI_API_KEY="sk-..."
```

## Build Wikipedia Knowledge Base

Create `topics.txt`:
```
Python_(programming_language)
Machine_learning
Artificial_intelligence
```

Ingest Wikipedia:
```bash
ct build-wiki --topics-file topics.txt --limit 3 --verbose
```

This creates `data/wiki/segment_*.is` files.

## Search

```bash
ct search "What is machine learning?"
```

Output:
```
[TOP 1] score=0.6407 text_seg=2 emb_seg=3
  text_path=data/wiki/segment_2.is
  Machine learning (ML) is a field of study in artificial intelligence...
```

## Chat (Interactive)

```bash
ct chat
```

Type your questions. The system retrieves relevant context from `data/wiki/` and remembers your chat in `data/chat/`.

## That's It!

- **Wikipedia data** → `data/wiki/segment_*.is`
- **Chat memory** → `data/chat/segment_*.is`
- **Search** → finds relevant segments, shows scores
- **Chat** → interactive Q&A with memory

## Next Steps

See [README.md](README.md) for:
- Python API usage
- Custom document ingestion
- Benchmark tools
- Advanced configuration
