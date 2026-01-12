# ContextTape Organization Guide

This document clarifies the **complete organization** of the ContextTape package and explains what each directory does.

## 📦 Package Structure (Clean)

After running `cleanup_stores.sh`, you should have this clean structure:

```
contexttape/
├── src/                       ← SOURCE CODE (the actual package)
│   └── contexttape/
│       ├── __init__.py       ← Package entry point
│       ├── storage.py        ← Core storage engine
│       ├── embed.py          ← Embedding generation
│       ├── search.py         ← Search algorithms
│       ├── ingest.py         ← Content ingestion
│       ├── client.py         ← Client API
│       ├── energy.py         ← Energy monitoring
│       └── cli.py            ← Command-line interface
│
├── tests/                     ← TEST SUITE
│   ├── test_storage.py       ← 41 storage tests
│   └── test_integration.py   ← 14 integration tests
│
├── examples/                  ← USAGE EXAMPLES
│   ├── quickstart.py         ← 7 basic examples
│   ├── advanced_usage.py     ← 7 advanced patterns
│   ├── tutorial.py           ← 5 step-by-step tutorials
│   └── benchmark.py          ← Performance testing
│
├── docs/                      ← DOCUMENTATION
│   ├── architecture.md       ← System design
│   ├── api_reference.md      ← API documentation
│   ├── performance.md        ← Performance guide
│   └── deployment.md         ← Deployment guide
│
├── .github/                   ← CI/CD CONFIGURATION
│   └── workflows/
│       └── ci.yml            ← GitHub Actions tests
│
├── bench/                     ← BENCHMARKING CODE
│   └── (performance tests)
│
├── scripts/                   ← UTILITY SCRIPTS
│   └── (development tools)
│
├── sample_corpus/             ← SAMPLE DATA
│   └── (example documents)
│
├── sources/                   ← RESEARCH/NOTES
│   └── (development notes)
│
├── README.md                  ← Main documentation
├── QUICK_REFERENCE.md         ← Quick reference card
├── PROJECT_STRUCTURE.md       ← This file
├── CONTRIBUTING.md            ← Contribution guide
├── CODE_OF_CONDUCT.md         ← Community standards
├── pyproject.toml             ← Package configuration
├── LICENSE                    ← MIT License
├── .gitignore                 ← Exclude patterns
├── cleanup_stores.sh          ← Store cleanup script
└── verify_setup.py            ← System verification
```

## 🚫 What You Should NOT See (After Cleanup)

These directories are **temporary user data** created by running examples/tests:

```
❌ batch_store/               (created by examples/quickstart.py)
❌ chat_ts/                   (created by chat examples)
❌ embedding_store/           (created by examples)
❌ multi_chat/                (created by multi-store examples)
❌ multi_wiki/                (created by multi-store examples)
❌ quickstart_store/          (created by examples/quickstart.py)
❌ search_store/              (created by search examples)
❌ stats_store/               (created by stats examples)
❌ wiki_store/                (created by Wikipedia examples)
❌ tutorial_*/                (created by examples/tutorial.py)
❌ hierarchy/                 (created by hierarchical examples)
```

**These are NOT part of the package—they are runtime-generated user data!**

## 🧹 How to Clean Up

Run the cleanup script anytime:

```bash
cd contexttape
bash cleanup_stores.sh
```

Or skip confirmation:

```bash
bash cleanup_stores.sh -y
```

## 🔍 Understanding the Distinction

### Source Code vs User Data

| Type | Location | In Git? | Purpose |
|------|----------|---------|---------|
| **Source Code** | `src/contexttape/` | ✅ Yes | The actual package code |
| **Tests** | `tests/` | ✅ Yes | Automated test suite |
| **Examples** | `examples/` | ✅ Yes | Usage demonstrations |
| **Docs** | `docs/`, `*.md` | ✅ Yes | Documentation |
| **Config** | `pyproject.toml`, `.gitignore` | ✅ Yes | Package configuration |
| **User Data** | `*_store/`, `*_ts/` | ❌ No | Runtime-generated stores |

### Why Are Store Directories Created?

When you run examples, they create temporary directories to demonstrate the system:

```python
# examples/quickstart.py line 15
store = SegmentedStore("quickstart_store")  # Creates quickstart_store/ directory
store.append_text_with_embedding("Hello", [0.1, 0.2, ...])
```

This is **expected behavior**—the system creates these directories to store your data.

## 📋 Directory Purposes

### Essential (Always Present)

- **`src/contexttape/`** — The actual Python package with 8 core modules
- **`tests/`** — 55 tests ensuring everything works
- **`examples/`** — 20+ examples showing how to use the package
- **`docs/`** — Comprehensive documentation

### Configuration (Always Present)

- **`pyproject.toml`** — Package metadata, dependencies, build config
- **`.gitignore`** — Prevents committing temporary stores
- **`LICENSE`** — MIT License
- **`README.md`** — Main documentation

### Development (Always Present)

- **`.github/workflows/`** — CI/CD with GitHub Actions
- **`bench/`** — Performance benchmarking code
- **`scripts/`** — Development utilities
- **`verify_setup.py`** — System verification script
- **`cleanup_stores.sh`** — Store cleanup utility

### Sample Data (Optional, Can Delete)

- **`sample_corpus/`** — Example documents for testing
- **`sources/`** — Research notes and development materials

### Temporary (Created by Examples)

- **`*_store/`** — User data stores (not source code)
- **`*_ts/`** — Temporary stores
- **`tutorial_*/`** — Created by tutorial examples
- **`multi_*/`** — Created by multi-store examples

## 🎯 For New Users

### First Time Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/NuTerraLabs/contexttape.git
   cd contexttape
   ```

2. **Install the package**
   ```bash
   pip install -e .
   ```

3. **Verify installation**
   ```bash
   python verify_setup.py
   ```

4. **Run examples**
   ```bash
   python examples/quickstart.py
   ```

5. **You'll see new directories** like `quickstart_store/`—this is normal!

6. **Clean up when done**
   ```bash
   bash cleanup_stores.sh
   ```

### Day-to-Day Usage

When you use ContextTape in your projects, you'll create your own store directories:

```python
from contexttape import SegmentedStore

# This creates "my_project_store/" in your current directory
store = SegmentedStore("my_project_store")
```

**Recommendation:** Keep your production stores in a dedicated directory:

```python
# Better organization
store = SegmentedStore("data/knowledge_base")
store = SegmentedStore("data/chat_history")
store = SegmentedStore("data/embeddings")
```

## 🗂️ Recommended Project Organization

When building applications with ContextTape:

```
my_app/
├── src/                      ← Your application code
│   └── app.py
├── data/                     ← Your ContextTape stores
│   ├── knowledge_base/       ← Store 1
│   ├── chat_history/         ← Store 2
│   └── embeddings/           ← Store 3
├── requirements.txt          ← Include: contexttape>=0.5.0
└── README.md
```

## 🔒 Git Configuration

The `.gitignore` file excludes all temporary stores:

```gitignore
# User data stores (runtime-generated)
*_store/
*_ts/
tutorial_*/
multi_*/
hierarchy/

# Python artifacts
__pycache__/
*.pyc
.pytest_cache/

# Build artifacts
dist/
build/
*.egg-info/
```

This ensures:
- ✅ Source code IS committed
- ✅ Tests ARE committed
- ✅ Documentation IS committed
- ❌ User data stores are NOT committed

## ❓ Common Questions

### Q: Why do I see `*_store` directories?
**A:** Examples create them to demonstrate the system. They're user data, not source code.

### Q: Should I commit `quickstart_store/` to git?
**A:** No—it's in `.gitignore`. Run `cleanup_stores.sh` to remove it.

### Q: Are these directories part of the package?
**A:** No—they're created by running examples. The package is in `src/contexttape/`.

### Q: How do I prevent creating these directories?
**A:** Don't run the examples, or clean up afterward with `cleanup_stores.sh`.

### Q: Will deleting them break anything?
**A:** No—they'll be recreated when you run examples again.

### Q: Where's the actual package code?
**A:** `src/contexttape/` contains all 8 Python modules (~2,000 lines of code).

### Q: What if I want to keep some stores?
**A:** Move them to a `data/` directory:
```bash
mkdir data
mv my_important_store data/
bash cleanup_stores.sh  # Removes temporary stores, keeps data/
```

## 📊 Size Reference

| Component | Size | Files |
|-----------|------|-------|
| Source code | ~500 KB | 8 Python files |
| Tests | ~100 KB | 2 test files |
| Examples | ~50 KB | 4 example files |
| Documentation | ~200 KB | 10 markdown files |
| **Total (clean)** | **~850 KB** | **~30 files** |
| Temporary stores | Varies | Created by examples |

## 🚀 Next Steps

1. **Read the main README**: [README.md](README.md)
2. **Try the quickstart**: `python examples/quickstart.py`
3. **Review the API**: [docs/api_reference.md](docs/api_reference.md)
4. **Run the tests**: `pytest tests/ -v`
5. **Clean up**: `bash cleanup_stores.sh`

---

**Remember:** Focus on `src/`, `tests/`, `examples/`, and `docs/`. Everything else is either configuration or temporary user data.
