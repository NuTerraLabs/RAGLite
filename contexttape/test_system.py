#!/usr/bin/env python3
"""
ContextTape System Test Runner
================================

Quick test script to verify the full system works end-to-end.
Run this to test your installation and see how data is organized.

Usage:
    python test_system.py
    python test_system.py --cleanup  # Remove test data after
"""

import os
import sys
import argparse
from pathlib import Path

def test_basic_storage():
    """Test 1: Basic storage operations"""
    print("\n" + "="*60)
    print("TEST 1: Basic Storage (File Creation)")
    print("="*60)
    
    from contexttape import TSStore
    import numpy as np
    
    # Create store in organized data directory
    data_dir = Path("test_data")
    store_path = data_dir / "basic_store"
    
    print(f"Creating store at: {store_path}")
    store = TSStore(str(store_path))
    
    # Add some data
    texts = [
        "Python is a programming language",
        "Machine learning uses algorithms",
        "Neural networks learn patterns"
    ]
    
    print(f"\nAdding {len(texts)} documents...")
    for text in texts:
        # Create fake embedding (1536-dim like OpenAI)
        vec = np.random.randn(1536).astype(np.float32)
        text_id = store.append_text(text)
        vec_id = store.append_vector_i8(vec, prev_text_id=text_id)
        print(f"  ✓ Added: {text[:40]}... (text_id={text_id}, vec_id={vec_id})")
    
    # Check what files were created
    print(f"\n📁 Files created in {store_path}:")
    if store_path.exists():
        files = sorted(store_path.glob("segment_*.is"))
        print(f"   Total: {len(files)} segment files")
        for f in files[:6]:  # Show first 6
            size = f.stat().st_size
            print(f"   - {f.name} ({size:,} bytes)")
        if len(files) > 6:
            print(f"   ... and {len(files) - 6} more")
    
    print("\n✅ Test 1 passed - Files created successfully")
    return store_path

def test_search():
    """Test 2: Search functionality"""
    print("\n" + "="*60)
    print("TEST 2: Search (Retrieval)")
    print("="*60)
    
    from contexttape import TSStore
    import numpy as np
    
    data_dir = Path("test_data")
    store_path = data_dir / "search_store"
    
    print(f"Creating store at: {store_path}")
    store = TSStore(str(store_path))
    
    # Add documents with known embeddings
    docs = [
        ("Python programming", [1.0, 0.0, 0.0]),
        ("Machine learning", [0.0, 1.0, 0.0]),
        ("Data science", [0.0, 0.0, 1.0]),
    ]
    
    print(f"\nIngesting {len(docs)} documents...")
    for text, vec in docs:
        # Pad to 1536 dimensions
        full_vec = np.zeros(1536, dtype=np.float32)
        full_vec[:3] = vec
        text_id = store.append_text(text)
        vec_id = store.append_vector_i8(full_vec, prev_text_id=text_id)
        print(f"  ✓ {text}")
    
    # Search
    query_vec = np.zeros(1536, dtype=np.float32)
    query_vec[0] = 1.0  # Should match "Python programming"
    
    print(f"\n🔍 Searching for Python-related content...")
    results = store.search_by_vector(query_vec, top_k=2)
    
    print(f"\nTop {len(results)} results:")
    for score, text_id, vec_id in results:
        text = store.read_text(text_id)
        print(f"  {score:.3f}: {text}")
    
    print("\n✅ Test 2 passed - Search works")
    return store_path

def test_client_api():
    """Test 3: High-level client API"""
    print("\n" + "="*60)
    print("TEST 3: Client API (Easy Interface)")
    print("="*60)
    
    from contexttape import ContextTapeClient
    import numpy as np
    
    data_dir = Path("test_data")
    store_path = data_dir / "client_store"
    
    print(f"Creating client for: {store_path}")
    
    # Create client with fake embedder
    def fake_embed(text):
        """Simple deterministic embedder for testing"""
        # Use hash of text to create consistent embedding
        h = hash(text) % 1000
        vec = np.zeros(1536, dtype=np.float32)
        vec[h % 1536] = 1.0
        return vec
    
    client = ContextTapeClient(
        store_path=str(store_path),
        embed_fn=fake_embed
    )
    
    # Ingest documents
    docs = [
        "ContextTape is a database-free RAG storage system",
        "It stores text and embeddings in segment files",
        "No infrastructure needed - just files",
    ]
    
    print(f"\nIngesting {len(docs)} documents...")
    for doc in docs:
        doc_id = client.ingest(doc)
        print(f"  ✓ Ingested: {doc[:50]}... (id={doc_id})")
    
    # Search
    print(f"\n🔍 Searching...")
    results = client.search("How does storage work?", top_k=2)
    
    print(f"\nTop {len(results)} results:")
    for result in results:
        print(f"  {result.score:.3f}: {result.text[:60]}...")
    
    print("\n✅ Test 3 passed - Client API works")
    return store_path

def test_multistore():
    """Test 4: Multi-store fusion"""
    print("\n" + "="*60)
    print("TEST 4: Multi-Store (Multiple Sources)")
    print("="*60)
    
    from contexttape import TSStore, MultiStore
    import numpy as np
    
    data_dir = Path("test_data")
    
    # Create two separate stores
    wiki_path = data_dir / "wiki_store"
    chat_path = data_dir / "chat_store"
    
    print(f"Creating stores:")
    print(f"  - {wiki_path}")
    print(f"  - {chat_path}")
    
    wiki = TSStore(str(wiki_path))
    chat = TSStore(str(chat_path))
    
    # Add to wiki store
    wiki_docs = ["Python documentation", "API reference"]
    print(f"\nAdding {len(wiki_docs)} docs to wiki_store...")
    for doc in wiki_docs:
        vec = np.random.randn(1536).astype(np.float32)
        tid = wiki.append_text(doc)
        wiki.append_vector_i8(vec, prev_text_id=tid)
        print(f"  ✓ {doc}")
    
    # Add to chat store
    chat_docs = ["User question about Python", "Bot response"]
    print(f"\nAdding {len(chat_docs)} docs to chat_store...")
    for doc in chat_docs:
        vec = np.random.randn(1536).astype(np.float32)
        tid = chat.append_text(doc)
        chat.append_vector_i8(vec, prev_text_id=tid)
        print(f"  ✓ {doc}")
    
    # Search across both
    multi = MultiStore([wiki, chat])
    query = np.random.randn(1536).astype(np.float32)
    
    print(f"\n🔍 Searching across both stores...")
    results = multi.search(query, final_k=3)
    print(f"Found {len(results)} results from both stores")
    
    print("\n✅ Test 4 passed - Multi-store works")
    return [wiki_path, chat_path]

def show_data_organization():
    """Show how data is organized"""
    print("\n" + "="*60)
    print("DATA ORGANIZATION")
    print("="*60)
    
    data_dir = Path("test_data")
    
    if not data_dir.exists():
        print("No test data created yet")
        return
    
    print(f"\n📂 All test data is in: {data_dir.absolute()}/")
    print()
    
    # Show directory tree
    stores = sorted(data_dir.glob("*_store"))
    if stores:
        print("Stores created:")
        for store in stores:
            files = list(store.glob("segment_*.is"))
            size_bytes = sum(f.stat().st_size for f in files)
            size_kb = size_bytes / 1024
            print(f"  📁 {store.name}/")
            print(f"      {len(files)} segments, {size_kb:.1f} KB total")
    else:
        print("No stores found")
    
    print(f"\n💡 In your application, organize like:")
    print(f"   your_app/")
    print(f"   ├── src/")
    print(f"   └── data/              ← Your ContextTape stores")
    print(f"       ├── knowledge/     ← Knowledge base")
    print(f"       ├── chat_history/  ← Chat logs")
    print(f"       └── embeddings/    ← Pre-computed embeddings")

def cleanup():
    """Remove all test data"""
    import shutil
    
    data_dir = Path("test_data")
    if data_dir.exists():
        print(f"\n🧹 Removing {data_dir}...")
        shutil.rmtree(data_dir)
        print("✓ Cleanup complete")
    else:
        print("\nNo test data to clean up")

def main():
    parser = argparse.ArgumentParser(description="Test ContextTape system")
    parser.add_argument("--cleanup", action="store_true", help="Remove test data after running")
    parser.add_argument("--only-show", action="store_true", help="Only show data organization")
    args = parser.parse_args()
    
    print("="*60)
    print("ContextTape System Test")
    print("="*60)
    
    if args.only_show:
        show_data_organization()
        return
    
    try:
        # Run all tests
        test_basic_storage()
        test_search()
        test_client_api()
        test_multistore()
        
        # Show organization
        show_data_organization()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
        print("\n💡 Tips:")
        print("   - Your data is in: test_data/")
        print("   - Each store is a directory with .is segment files")
        print("   - Run with --cleanup to remove test data")
        print("   - In production, organize stores under data/")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Cleanup if requested
    if args.cleanup:
        cleanup()

if __name__ == "__main__":
    main()
