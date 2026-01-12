#!/usr/bin/env python3
"""
Example: Using ContextTape as an installed pip package
=======================================================

This demonstrates what happens when you install contexttape
via pip and use it in your project.
"""

# After: pip install contexttape
from contexttape import TSStore
import numpy as np

print("="*60)
print("WHERE DOES YOUR DATA GET CREATED?")
print("="*60)
print()

# Example 1: Basic usage (data in current directory)
print("1. Basic usage - data in current directory:")
print()
print("   Your code:")
print("   store = TSStore('my_store')")
print()
print("   Creates:")
print("   ./my_store/")
print("   ├── segment_0.is")
print("   ├── segment_1.is")
print("   └── ...")
print()

# Example 2: Recommended - organized structure
print("2. RECOMMENDED - organized structure:")
print()
print("   Your project:")
print("   my_app/")
print("   ├── app.py              ← Your code")
print("   ├── requirements.txt    ← Lists: contexttape>=0.5.0")
print("   └── data/               ← Create this!")
print()
print("   In app.py:")
print("   store = TSStore('data/knowledge')")
print()
print("   Creates:")
print("   my_app/")
print("   └── data/")
print("       └── knowledge/")
print("           ├── segment_0.is")
print("           ├── segment_1.is")
print("           └── ...")
print()

# Example 3: Absolute paths
print("3. Using absolute paths:")
print()
print("   Your code:")
print("   store = TSStore('/var/app/data/store')")
print()
print("   Creates:")
print("   /var/app/data/store/")
print("   ├── segment_0.is")
print("   └── ...")
print()

# Live demonstration
print("="*60)
print("LIVE DEMO - Creating a store right now:")
print("="*60)
print()

import os
print(f"Current directory: {os.getcwd()}")
print()

# Create store in demo_data/
store = TSStore("demo_data/example_store")
print(f"✓ Created store at: demo_data/example_store/")
print()

# Add some data
text_id = store.append_text("Hello from ContextTape!")
vec = np.random.randn(128).astype(np.float32)
vec_id = store.append_vector_i8(vec, prev_text_id=text_id)

print(f"✓ Added text (segment_id={text_id})")
print(f"✓ Added vector (segment_id={vec_id})")
print()

# Show what was created
import glob
files = sorted(glob.glob("demo_data/example_store/segment_*.is"))
print(f"Files created:")
for f in files:
    size = os.path.getsize(f)
    print(f"  - {os.path.basename(f)} ({size} bytes)")
print()

print("="*60)
print("KEY POINTS:")
print("="*60)
print()
print("1. ContextTape is installed in: site-packages/contexttape/")
print("   (the package code)")
print()
print("2. Your DATA is created in: wherever you specify")
print("   TSStore('path') creates: ./path/")
print()
print("3. RECOMMENDED: Organize under data/")
print("   - Easy to .gitignore")
print("   - Clear separation")
print("   - Easy to backup")
print()
print("4. In .gitignore, add:")
print("   data/")
print("   demo_data/")
print("   *_store/")
print()

# Cleanup
import shutil
if os.path.exists("demo_data"):
    shutil.rmtree("demo_data")
    print("✓ Cleaned up demo_data/")
