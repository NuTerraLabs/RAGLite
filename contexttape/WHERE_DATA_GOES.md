# Where Does Your Data Get Created?

## TL;DR

**Data is created wherever you specify, relative to where your script runs.**

```python
from contexttape import ISStore

# This creates: ./my_store/ in your current directory
store = ISStore("my_store")

# This creates: ./data/knowledge/ in your current directory  
store = ISStore("data/knowledge")

# This creates at absolute path
store = ISStore("/var/app/data/store")
```

---

## The Two Parts

### 1. **Package Code** (installed via pip)

When you run `pip install contexttape`, the **package code** gets installed:

```
/path/to/python/site-packages/
└── contexttape/              ← Package installation
    ├── __init__.py
    ├── storage.py
    ├── embed.py
    └── ...
```

**This is read-only code you import from.**

### 2. **Your Data** (created when you use it)

When you **use** ContextTape in your project, **data gets created**:

```python
from contexttape import ISStore

store = ISStore("my_data")  # ← Creates directory HERE
```

**This creates `my_data/` in your current working directory.**

---

## Real-World Example

### Your Project Structure

```
my_rag_app/                    ← Your project
├── venv/                      ← Virtual environment
│   └── lib/python3.11/
│       └── site-packages/
│           └── contexttape/   ← Package installed here (pip install)
├── app.py                     ← Your application code
├── requirements.txt           ← Lists: contexttape>=0.5.0
└── data/                      ← YOU create this
```

### Your Code (app.py)

```python
from contexttape import ISStore

# Option 1: Relative path (creates in current directory)
store = ISStore("data/knowledge")

# Option 2: Absolute path
store = ISStore("/home/user/my_rag_app/data/knowledge")
```

### What Gets Created

```
my_rag_app/
├── app.py
├── data/                      ← Created by you
│   └── knowledge/             ← Created by ISStore("data/knowledge")
│       ├── segment_0.is       ← Created when you add data
│       ├── segment_1.is
│       └── ...
└── venv/
    └── lib/python3.11/
        └── site-packages/
            └── contexttape/   ← Package (read-only)
```

---

## Where Exactly?

The store directory is created **relative to your current working directory** when you run your script.

```bash
# If you run from here:
cd /home/user/my_rag_app
python app.py

# And your code has:
store = ISStore("data/knowledge")

# Then data is created at:
/home/user/my_rag_app/data/knowledge/
```

---

## Recommended Setup

### Step 1: Create Your Project

```bash
mkdir my_rag_app
cd my_rag_app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install contexttape
pip install contexttape
```

### Step 2: Create Data Directory

```bash
mkdir data
```

### Step 3: Create .gitignore

```bash
cat > .gitignore << EOF
# Python
venv/
__pycache__/
*.pyc

# ContextTape data stores
data/
*_store/
EOF
```

### Step 4: Write Your Code

```python
# app.py
from contexttape import ISStore
import numpy as np

def main():
    # All stores under data/
    knowledge_store = ISStore("data/knowledge")
    chat_store = ISStore("data/chat_history")
    cache_store = ISStore("data/cache")
    
    # Add some data
    text_id = knowledge_store.append_text("Hello world")
    vec = np.random.randn(1536).astype(np.float32)
    vec_id = knowledge_store.append_vector_i8(vec, prev_text_id=text_id)
    
    print(f"✓ Created data in data/knowledge/")
    print(f"  - segment_{text_id}.is (text)")
    print(f"  - segment_{vec_id}.is (vector)")

if __name__ == "__main__":
    main()
```

### Step 5: Run It

```bash
python app.py
```

### Result

```
my_rag_app/
├── app.py
├── .gitignore
├── data/                      ← Your data
│   ├── knowledge/
│   │   ├── segment_0.is
│   │   ├── segment_1.is
│   │   └── ...
│   ├── chat_history/
│   └── cache/
├── requirements.txt
└── venv/                      ← Package installed here
    └── lib/python3.11/
        └── site-packages/
            └── contexttape/
```

---

## Common Scenarios

### Scenario 1: Web Application

```python
# config.py
import os
from pathlib import Path

# Get project root
BASE_DIR = Path(__file__).parent

# All stores under data/
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

STORES = {
    "knowledge": str(DATA_DIR / "knowledge"),
    "users": str(DATA_DIR / "users"),
    "cache": str(DATA_DIR / "cache"),
}

# app.py
from contexttape import ISStore
from config import STORES

knowledge = ISStore(STORES["knowledge"])
users = ISStore(STORES["users"])
```

**Data location:** `{your_app}/data/knowledge/`, `{your_app}/data/users/`, etc.

### Scenario 2: Docker Container

```dockerfile
FROM python:3.11

WORKDIR /app

# Install package
RUN pip install contexttape

# Create data directory
RUN mkdir -p /app/data

# Copy your code
COPY app.py /app/

# Run
CMD ["python", "app.py"]
```

```python
# app.py in container
store = ISStore("/app/data/knowledge")
# Creates: /app/data/knowledge/ inside container
```

**Mount volume for persistence:**
```bash
docker run -v /host/data:/app/data my_app
```

### Scenario 3: Jupyter Notebook

```python
# In notebook
from contexttape import ISStore
import os

# Check where you are
print(f"Working directory: {os.getcwd()}")
# Output: /home/user/notebooks

# Create store
store = ISStore("notebook_data/experiments")
# Creates: /home/user/notebooks/notebook_data/experiments/
```

---

## Key Points

| Question | Answer |
|----------|--------|
| **Where is the package?** | `site-packages/contexttape/` (installed by pip) |
| **Where is my data?** | Wherever you specify: `ISStore("path")` |
| **Is data created automatically?** | Yes, directory created on first use |
| **Can I use absolute paths?** | Yes: `ISStore("/absolute/path")` |
| **Can I use relative paths?** | Yes: `ISStore("relative/path")` (from cwd) |
| **Should I commit data to git?** | No - add to .gitignore |
| **Can I move data?** | Yes - just move/copy the directory |
| **Can multiple stores exist?** | Yes - create multiple ISStore instances |

---

## Testing Where Data Goes

Run this to see for yourself:

```python
from contexttape import ISStore
import os

print(f"Current directory: {os.getcwd()}")

store = ISStore("test_store")
print(f"Store created at: {os.path.abspath('test_store')}")

# Check if it exists
print(f"Exists: {os.path.exists('test_store')}")
```

---

## Best Practices

### ✅ DO:

```python
# Organize under data/
store = ISStore("data/my_store")

# Use config for paths
from config import DATA_DIR
store = ISStore(f"{DATA_DIR}/my_store")

# Use pathlib for cross-platform
from pathlib import Path
data_dir = Path("data")
store = ISStore(str(data_dir / "my_store"))
```

### ❌ DON'T:

```python
# Don't pollute current directory
store = ISStore("my_store")  # Creates ./my_store in cwd

# Don't use hardcoded absolute paths
store = ISStore("/home/john/stores")  # Not portable

# Don't commit stores to git
# (add to .gitignore instead)
```

---

## Summary

1. **Package installation** (pip): → `site-packages/contexttape/`
2. **Your data** (runtime): → wherever you specify with `ISStore("path")`
3. **Recommended**: Organize all stores under `data/` directory
4. **Add to .gitignore**: `data/` and `*_store/`

**The package is the code. Your data is wherever you create it.**
