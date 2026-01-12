# ContextTape: Package vs Data Locations

## Visual Guide

```
┌─────────────────────────────────────────────────────────────┐
│                    YOUR PYTHON ENVIRONMENT                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  /path/to/python/site-packages/                            │
│  └── contexttape/              ← pip install puts code here│
│      ├── __init__.py                                        │
│      ├── storage.py           ← Package code (READ ONLY)   │
│      ├── embed.py                                           │
│      └── ...                                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘

                            ↓ import

┌─────────────────────────────────────────────────────────────┐
│                      YOUR PROJECT                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  my_rag_app/                  ← Your working directory     │
│  ├── app.py                   ← Your code                  │
│  │   from contexttape import ISStore                       │
│  │   store = ISStore("data/knowledge")  ← Creates below   │
│  │                                                          │
│  ├── requirements.txt                                       │
│  │   contexttape>=0.5.0                                    │
│  │                                                          │
│  └── data/                    ← YOU create this            │
│      └── knowledge/           ← ISStore creates this       │
│          ├── segment_0.is     ← Your data files           │
│          ├── segment_1.is                                  │
│          └── ...              ← WRITABLE, YOUR DATA        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## The Two Locations

### 1️⃣ Package Code (from pip)

```bash
pip install contexttape
```

**Goes to:** `site-packages/contexttape/`

- This is the **library code**
- Read-only (managed by pip)
- Same for all your projects
- Import from here: `from contexttape import ISStore`

### 2️⃣ Your Data (from usage)

```python
store = ISStore("data/knowledge")
```

**Creates:** `{current_directory}/data/knowledge/`

- This is **your data**
- Writable (you control it)
- Different for each project
- Lives in your project directory

## Real Example

```bash
# 1. Install the package
cd ~/my_projects/rag_app
pip install contexttape

# Where did it go?
# → ~/.local/lib/python3.11/site-packages/contexttape/

# 2. Use in your code
cat > app.py << 'PYEOF'
from contexttape import ISStore

store = ISStore("data/knowledge")
store.append_text("Hello")
PYEOF

# 3. Run it
python app.py

# Where did DATA go?
# → ~/my_projects/rag_app/data/knowledge/segment_0.is
```

## Common Confusion

❌ **Wrong thinking:** "I installed contexttape, so my data goes in site-packages/"

✅ **Correct:** "I installed contexttape (package code). My data goes wherever I create the store."

```python
# Package location: site-packages/contexttape/ (pip)
# Data location: wherever you run this from

store = ISStore("my_data")  # Creates: ./my_data/
```

## Quick Test

Run this to see for yourself:

```bash
cd ~
python3 << 'EOF'
import sys
import os

# Where is the package?
import contexttape
print(f"Package location: {contexttape.__file__}")

# Where will data go?
from contexttape import ISStore
store = ISStore("test_store")
print(f"Data location: {os.path.abspath('test_store')}")

# Cleanup
import shutil
shutil.rmtree("test_store")
