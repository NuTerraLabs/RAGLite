# Testing ContextTape with Wikipedia

## Quick Start

### Option 1: Using the CLI (Easiest)

```bash
cd /home/doom/RAGLite/contexttape

# 1. Create a topics file
cat > topics.txt << 'EOF'
Python_(programming_language)
Machine_learning
Artificial_intelligence
EOF

# 2. Build Wikipedia store
ct build-wiki --topics-file topics.txt --out-dir data/wiki --limit 5 --verbose

# 3. Search it
ct search "What is Python?" --wiki-dir data/wiki --topk 3

# 4. Check what was created
ls -lh data/wiki/
```

### Option 2: Using Python API

```python
# test_wiki_simple.py
from contexttape import ISStore, embed_text_1536, get_client
from contexttape.ingest_wiki import fetch_wiki_page

# Create store
store = ISStore("data/my_wiki")
client = get_client()

# Fetch and ingest pages
topics = ["Python_(programming_language)", "Machine_learning"]
for topic in topics:
    text = fetch_wiki_page(topic)
    vec = embed_text_1536(client, text[:8000])  # Limit to 8K chars
    text_id = store.append_text(text)
    vec_id = store.append_vector_i8(vec, prev_text_id=text_id)
    print(f"✓ {topic}: text={text_id}, vec={vec_id}")

# Search
query_vec = embed_text_1536(client, "What is machine learning?")
results = store.search_by_vector(query_vec, top_k=3)

for score, text_id, vec_id in results:
    text = store.read_text(text_id)
    print(f"{score:.4f}: {text[:100]}...")
```

## What You Just Did

### Files Created

```
data/wiki/
├── segment_0.is    # Python article (text)
├── segment_1.is    # Python embedding (vector)
├── segment_2.is    # Machine learning article (text)
└── segment_3.is    # Machine learning embedding (vector)
```

### How It Works

1. **Fetch**: Downloads Wikipedia page HTML
2. **Parse**: Extracts text content (strips tables, scripts, etc.)
3. **Embed**: Creates 1536-dim vector using OpenAI
4. **Store**: Saves as paired `.is` files
5. **Search**: Compares query vector to stored vectors

## Full Example Session

```bash
# Install if needed
pip install contexttape

# Set API key
export OPENAI_API_KEY="sk-..."

# Build from topics file
ct build-wiki --topics-file topics.txt --out-dir data/wiki --limit 10 --verbose

# See stats
ct stat --wiki-dir data/wiki

# Search
ct search "quantum computing" --wiki-dir data/wiki --topk 5

# Search with verbose output (shows file paths)
ct search "neural networks" --wiki-dir data/wiki --topk 3 --verbose

# Chat (interactive)
ct chat --wiki-dir data/wiki --topk 5
```

## Search Output Explained

```
[TOP 1] src=wiki score=0.6407 text_seg=2 emb_seg=3
  text_path=data/wiki_cli/segment_2.is
  vec_path=data/wiki_cli/segment_3.is
  preview: Study of algorithms that improve...
```

- **score**: Cosine similarity (0-1, higher is better)
- **text_seg**: Segment ID of the text
- **emb_seg**: Segment ID of the embedding
- **text_path**: Actual file containing the text
- **vec_path**: Actual file containing the vector

## Advanced: Multi-Store Search

```bash
# Build multiple stores
ct build-wiki --topics-file ai_topics.txt --out-dir data/ai_wiki
ct build-wiki --topics-file math_topics.txt --out-dir data/math_wiki

# Search across both (in Python)
from contexttape import ISStore, MultiStore, embed_text_1536, get_client

ai_store = ISStore("data/ai_wiki")
math_store = ISStore("data/math_wiki")
multi = MultiStore([ai_store, math_store])

client = get_client()
query_vec = embed_text_1536(client, "linear algebra in AI")
results = multi.search(query_vec, final_k=10)

for store_path, score, text_id, vec_id in results:
    print(f"{score:.4f} from {store_path}")
```

## Test Scripts Included

### 1. `test_wikipedia.py`
Complete test with Wikipedia ingestion and search

```bash
python test_wikipedia.py
```

### 2. `test_system.py`
Full system test (basic, search, client, multi-store)

```bash
python test_system.py
```

## Topics File Format

```
# topics.txt - one per line, use Wikipedia URL format
Python_(programming_language)
Machine_learning
Artificial_intelligence
Deep_learning
# Comments start with #
```

## Common Commands

```bash
# Build wiki store
ct build-wiki --topics-file topics.txt --out-dir data/wiki

# Search
ct search "your query" --wiki-dir data/wiki --topk 5

# Stats
ct stat --wiki-dir data/wiki

# Interactive chat
ct chat --wiki-dir data/wiki

# Ingest local files
ct ingest-path ./docs --out-dir data/docs

# Benchmark
ct bench --wiki-dir data/wiki --topk 5
```

## Cleanup

```bash
# Remove test data
rm -rf data/wiki_test data/wiki_cli data/my_wiki

# Or use the cleanup script
bash cleanup_stores.sh
```

## Troubleshooting

### No results found
- Check embeddings are being created (use `--verbose`)
- Verify OPENAI_API_KEY is set
- Try more documents (need at least a few for meaningful search)

### "count_tokens" error
- Fixed in latest version
- Make sure you have: `pip install -e .`

### Files not created
- Check permissions on data/ directory
- Verify path is correct: `ls -la data/`

## Performance

- **Ingestion**: ~2-3 pages/second (limited by OpenAI API)
- **Search**: ~50ms per query on 100 documents
- **Storage**: ~6KB per page (49KB text + 6KB vector int8)

## What's Happening Under the Hood

1. **Fetch** Wikipedia via requests + BeautifulSoup
2. **Clean** HTML → text (remove tables, scripts)
3. **Chunk** if >8K tokens (default: 800 token windows, 200 overlap)
4. **Embed** each chunk via OpenAI text-embedding-3-small
5. **Pool** chunk vectors → single vector (mean pooling)
6. **Quantize** float32 → int8 (4× compression)
7. **Write** paired segments: text `.is` + vector `.is`

## Next Steps

- Try with your own topics
- Combine wiki + chat stores
- Use multi-store for different knowledge domains
- Build RAG applications on top
