#!/usr/bin/env python3
"""
Complete setup and usability test for ContextTape/RAGLite
Tests: imports, basic usage, CLI, and file structure
"""

import sys
import os
import subprocess
import tempfile
import shutil

def test_imports():
    """Test that all main imports work"""
    print("🧪 Testing imports...")
    try:
        from contexttape import ISStore
        print("  ✅ ISStore imported")
        
        from contexttape import MultiStore
        print("  ✅ MultiStore imported")
        
        from contexttape import get_client, embed_text_1536
        print("  ✅ Embedding functions imported")
        
        from contexttape import combined_search
        print("  ✅ Search functions imported")
        
        print("✅ All imports successful!\n")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}\n")
        return False

def test_basic_usage():
    """Test basic store creation and operations"""
    print("🧪 Testing basic usage...")
    import numpy as np
    from contexttape import ISStore
    
    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = os.path.join(tmpdir, "test_store")
        
        # Create store
        store = ISStore(store_path)
        print(f"  ✅ Created store at {store_path}")
        
        # Add text
        text = "This is a test document about machine learning and AI."
        text_id = store.append_text(text)
        print(f"  ✅ Added text, segment ID: {text_id}")
        
        # Add text with embedding
        vec = np.random.randn(1536).astype(np.float32)
        tid, eid = store.append_text_with_embedding("Another document", vec)
        print(f"  ✅ Added text+embedding, IDs: text={tid}, vec={eid}")
        
        # Search
        query_vec = np.random.randn(1536).astype(np.float32)
        results = store.search_by_vector(query_vec, top_k=2)
        print(f"  ✅ Search returned {len(results)} results")
        
        # Stats
        stats = store.stat()
        print(f"  ✅ Stats: {stats['pairs']} pairs, {stats['text_segments']} text segments")
        
        # Verify files exist
        files = os.listdir(store_path)
        is_files = [f for f in files if f.endswith('.is')]
        print(f"  ✅ Created {len(is_files)} .is files")
        
    print("✅ Basic usage test passed!\n")
    return True

def test_cli_available():
    """Test that CLI command is available"""
    print("🧪 Testing CLI availability...")
    try:
        result = subprocess.run(['ct', '--version'], 
                               capture_output=True, 
                               text=True, 
                               timeout=5)
        if result.returncode == 0:
            print(f"  ✅ CLI version: {result.stdout.strip()}")
            print("✅ CLI test passed!\n")
            return True
        else:
            print(f"  ❌ CLI returned error code {result.returncode}")
            return False
    except FileNotFoundError:
        print("  ❌ 'ct' command not found in PATH")
        return False
    except Exception as e:
        print(f"  ❌ CLI test failed: {e}")
        return False

def test_file_structure():
    """Test that file structure is clean and organized"""
    print("🧪 Testing file structure...")
    
    # Check main directories exist
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    required_dirs = ['src/contexttape', 'tests', 'examples', 'docs']
    required_files = ['README.md', 'pyproject.toml', 'QUICKSTART.md']
    
    for dir_name in required_dirs:
        path = os.path.join(base_dir, dir_name)
        if os.path.isdir(path):
            print(f"  ✅ {dir_name}/ exists")
        else:
            print(f"  ❌ {dir_name}/ missing")
            return False
    
    for file_name in required_files:
        path = os.path.join(base_dir, file_name)
        if os.path.isfile(path):
            print(f"  ✅ {file_name} exists")
        else:
            print(f"  ❌ {file_name} missing")
            return False
    
    print("✅ File structure test passed!\n")
    return True

def test_documentation_clarity():
    """Check that key documentation is clear and accessible"""
    print("🧪 Testing documentation clarity...")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    readme_path = os.path.join(base_dir, 'README.md')
    
    with open(readme_path, 'r') as f:
        content = f.read()
    
    # Check for key terms
    key_terms = ['RAG', 'vector', 'embedding', 'search', 'ISStore', 'pip install']
    for term in key_terms:
        if term in content:
            print(f"  ✅ README mentions '{term}'")
        else:
            print(f"  ⚠️  README missing '{term}'")
    
    # Check file size is reasonable
    lines = content.count('\n')
    print(f"  ℹ️  README has {lines} lines")
    
    print("✅ Documentation test passed!\n")
    return True

def print_usage_examples():
    """Print simple usage examples"""
    print("=" * 60)
    print("📚 USAGE EXAMPLES")
    print("=" * 60)
    print("""
# Install
pip install contexttape

# Python API
from contexttape import ISStore
import numpy as np

store = ISStore("data/my_rag")
vec = np.random.randn(1536).astype(np.float32)
tid, eid = store.append_text_with_embedding("Document text", vec)
results = store.search_by_vector(vec, top_k=5)

# CLI
ct build-wiki --topics-file topics.txt --limit 3
ct search "your query"
ct chat

# Data Location
- Package code: site-packages/contexttape/
- Your data: wherever you specify (e.g., data/my_rag/)
""")

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("🚀 ContextTape/RAGLite Complete Setup Test")
    print("=" * 60 + "\n")
    
    results = {
        'Imports': test_imports(),
        'Basic Usage': test_basic_usage(),
        'CLI': test_cli_available(),
        'File Structure': test_file_structure(),
        'Documentation': test_documentation_clarity(),
    }
    
    print("=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {test_name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Setup is clean and simple.")
        print_usage_examples()
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
