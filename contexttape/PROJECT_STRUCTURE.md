# ContextTape Project Structure

This document explains the organization and purpose of every directory and key file in the ContextTape project.

## 📁 Directory Overview

```
contexttape/
├── 📦 src/                      # Source code (the actual package)
│   └── contexttape/            # Main package
│       ├── __init__.py         # Package exports & API
│       ├── storage.py          # Core storage engine
│       ├── embed.py            # Embedding utilities
│       ├── search.py           # Search & retrieval
│       ├── cli.py              # Command-line interface
│       ├── integrations.py     # Framework integrations
│       ├── benchmark.py        # Performance testing
│       ├── chat.py             # Chat memory
│       ├── config.py           # Configuration
│       ├── energy.py           # Energy monitoring
│       ├── ingest_*.py         # Data ingestion modules
│       ├── relevance.py        # Relevance scoring
│       └── utils.py            # Utilities
│
├── 🧪 tests/                    # Test suite
│   ├── test_storage.py         # Storage tests (41 tests)
│   └── test_integration.py     # Integration tests (14 tests)
│
├── 📚 examples/                 # Usage examples
│   ├── quickstart.py           # 7 basic examples
│   ├── advanced_usage.py       # 7 advanced patterns
│   ├── tutorial.py             # 5 step-by-step tutorials
│   └── comprehensive_examples.py # Original examples
│
├── 📖 docs/                     # Documentation
│   ├── index.md                # Main documentation
│   ├── quickstart.md           # Quick start guide
│   ├── cli.md                  # CLI reference
│   ├── python_api.md           # Python API docs
│   └── requirements-docs.txt   # Docs dependencies
│
├── 📊 sample_corpus/            # Sample data for testing
│   ├── doc.json                # Sample JSON document
│   ├── glossary.csv            # Sample CSV
│   ├── *.md                    # Sample markdown files
│   └── *.txt                   # Sample text files
│
├── 🔧 scripts/                  # Utility scripts
│   ├── gen_docs.py             # Generate documentation
│   └── topics.example.txt      # Example topics for wiki
│
├── ⚙️ .github/                  # GitHub configuration
│   └── workflows/
│       └── ci.yml              # CI/CD pipeline
│
├── 📋 Configuration Files       # Package configuration
│   ├── pyproject.toml          # Package metadata & dependencies
│   ├── pytest.ini              # Pytest configuration
│   ├── mkdocs.yml              # Documentation site config
│   ├── requirements.txt        # Python dependencies
│   └── .gitignore              # Git ignore rules
│
├── 📄 Documentation Files       # Project documentation
│   ├── README.md               # Main README
│   ├── CHANGELOG.md            # Version history
│   ├── LICENSE                 # MIT License
│   ├── CONTRIBUTING.md         # Contribution guide
│   ├── CODE_OF_CONDUCT.md      # Community guidelines
│   ├── QUICK_REFERENCE.md      # Quick reference card
│   └── ENHANCEMENT_SUMMARY.md  # Recent enhancements
│
└── 🛠️ Utility Scripts           # Development tools
    ├── verify_setup.py         # System verification
    └── seed_multimodal_corpus.py # Generate test data
```

## 📦 What Gets Created at Runtime

When you use ContextTape, it creates **data stores** as directories. These are NOT part of the package—they're user data:

### Store Directories (Created by Users/Examples)

These directories are **created dynamically** and contain your actual data:

```
<your_chosen_name>/          # A ContextTape store
├── segment_0.is             # Text segment
├── segment_1.is             # Vector segment (paired with segment_0)
├── segment_2.is             # Text segment
├── segment_3.is             # Vector segment (paired with segment_2)
└── ...                      # More segment pairs
```

**Common store names you might see:**
- `wiki_store/` - Wikipedia content (from examples)
- `chat_ts/` - Chat history (from examples)
- `my_knowledge_base/` - Your custom store
- Any name you choose when creating a store

**These directories:**
- ✅ Are created when you run `ISStore("directory_name")`
- ✅ Contain your actual data (text + embeddings)
- ✅ Should be in `.gitignore` (user data, not source code)
- ✅ Can be backed up, moved, or deleted independently
- ✅ Are portable—just copy the folder

## 🎯 What Each Component Does

### Core Source Code (`src/contexttape/`)

| File | Purpose | Key Functions |
|------|---------|---------------|
| `storage.py` | Core storage engine | `ISStore`, `MultiStore`, segment I/O |
| `embed.py` | Embedding generation | `embed_text_1536()`, OpenAI client |
| `search.py` | Search & retrieval | `combined_search()`, hybrid scoring |
| `cli.py` | Command-line tools | `ct` command with subcommands |
| `integrations.py` | Framework bridges | FastAPI, LangChain, LlamaIndex |
| `benchmark.py` | Performance testing | Latency, throughput, memory metrics |

### Examples (`examples/`)

| File | What It Shows | Use Case |
|------|---------------|----------|
| `quickstart.py` | Basic operations | New users, simple patterns |
| `advanced_usage.py` | Advanced patterns | Production use, complex scenarios |
| `tutorial.py` | Step-by-step guide | Learning the system |
| `comprehensive_examples.py` | Original examples | Legacy reference |

### Tests (`tests/`)

| File | Coverage | Tests |
|------|----------|-------|
| `test_storage.py` | Core storage | 41 tests (78% coverage) |
| `test_integration.py` | End-to-end workflows | 14 tests |

## 🔄 Data Flow

```
1. User creates store:
   ISStore("my_store") → creates my_store/ directory

2. User adds data:
   store.append_text_with_embedding(text, embedding)
   → creates segment_N.is files in my_store/

3. User searches:
   store.search_by_vector(query)
   → reads segment files, returns results

4. Data persists:
   my_store/ directory contains all data
   → Can be backed up, moved, shared
```

## 📊 Storage Format

Each `.ts` file is a binary segment:

```
[32-byte header][variable payload]
```

**Header contains:**
- Link to paired segment
- Data type (text/vector/JSON/blob)
- Payload length
- Vector dimension
- Quantization scale
- Timestamp

**Two files per document:**
- `segment_0.is` - UTF-8 text
- `segment_1.is` - Float32 or int8 quantized vector

## 🚀 Quick Start Reference

### Installing
```bash
pip install contexttape
```

### Creating a Store
```python
from contexttape import ISStore
store = ISStore("my_knowledge_base")  # Creates my_knowledge_base/ directory
```

### Adding Data
```python
store.append_text_with_embedding(
    "Your text here",
    embedding_vector,
    quantize=True  # 4x space savings
)
```

### Searching
```python
results = store.search_by_vector(query_vector, top_k=5)
for score, text_id, vec_id in results:
    print(store.read_text(text_id))
```

## 🧹 Cleaning Up

To remove all example/test stores:
```bash
# Remove all generated stores
rm -rf *_store/ *_ts/ tutorial_*/ multi_*/

# Keep only source code and configuration
git clean -fdx  # WARNING: Removes ALL untracked files
```

## 📦 Package vs User Data

| Type | Location | In Git? | Purpose |
|------|----------|---------|---------|
| **Package** | `src/contexttape/` | ✅ Yes | Source code |
| **Tests** | `tests/` | ✅ Yes | Test suite |
| **Examples** | `examples/` | ✅ Yes | Demo code |
| **Docs** | `docs/`, `*.md` | ✅ Yes | Documentation |
| **Config** | `pyproject.toml`, etc. | ✅ Yes | Package config |
| **User Stores** | `*_store/`, `*_ts/` | ❌ No | Your data |
| **Build** | `dist/`, `build/` | ❌ No | Generated |
| **Cache** | `__pycache__/`, `.pytest_cache/` | ❌ No | Temporary |

## 🎓 Learning Path

1. **Start here**: [README.md](README.md) - Overview & quick start
2. **Run examples**: `python examples/quickstart.py`
3. **Learn patterns**: `python examples/tutorial.py`
4. **API reference**: [docs/python_api.md](docs/python_api.md)
5. **Advanced**: `python examples/advanced_usage.py`
6. **Production**: [CONTRIBUTING.md](CONTRIBUTING.md)

## 🆘 Common Questions

**Q: Where is my data stored?**
A: In the directory you specify: `ISStore("my_store")` creates `./my_store/`

**Q: Why are there so many `*_store` directories?**
A: These are created by examples and tests. They're temporary—not part of the package.

**Q: Can I delete these directories?**
A: Yes! They're regenerated when you run examples. Your actual data is separate.

**Q: How do I back up my data?**
A: Just copy the store directory: `cp -r my_store/ backup/`

**Q: Where's the vector database?**
A: There isn't one! ContextTape uses files instead of a database.

## 📞 Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: info@nuterralabs.com
- **Documentation**: [docs/](docs/)

---

**Last Updated**: January 12, 2026
**Version**: 0.5.0
