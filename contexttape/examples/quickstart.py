"""
ContextTape Quickstart Examples
================================

Basic usage examples to get started with ContextTape.

NOTE: These examples create temporary store directories (*_store, *_ts).
      These are runtime-generated user data, NOT part of the package.
      Clean up afterward: bash cleanup_stores.sh
"""

import numpy as np
from contexttape import ISStore, MultiStore, embed_text_1536, get_client
import os

# ============================================================================
# Example 1: Basic Store Operations
# ============================================================================

def example_basic_store():
    """Create a store, add text, and retrieve it."""
    print("=" * 60)
    print("Example 1: Basic Store Operations")
    print("=" * 60)
    
    # Create a new store
    store = ISStore("./quickstart_store")
    
    # Add some text
    texts = [
        "Python is a high-level programming language.",
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are inspired by the human brain.",
    ]
    
    for i, text in enumerate(texts):
        text_id = store.append_text(text)
        print(f"Stored text {i}: ID={text_id}, Text='{text}'")
    
    # Read text back
    print("\nReading text back:")
    for i in range(len(texts)):
        retrieved = store.read_text(i)
        print(f"  ID {i}: {retrieved}")
    
    print()


# ============================================================================
# Example 2: Text with Embeddings
# ============================================================================

def example_text_with_embeddings():
    """Store text with vector embeddings for semantic search."""
    print("=" * 60)
    print("Example 2: Text with Embeddings")
    print("=" * 60)
    
    store = ISStore("./embedding_store")
    
    # For this example, we'll use random embeddings
    # In production, use OpenAI or another embedding model
    docs = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models learn patterns from data.",
        "Python is widely used in data science and AI.",
    ]
    
    print("Storing documents with embeddings:")
    for doc in docs:
        # Generate a random embedding (replace with real embeddings)
        embedding = np.random.randn(1536).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        
        # Store text with embedding (quantized to save space)
        text_id, vec_id = store.append_text_with_embedding(doc, embedding, quantize=True)
        print(f"  Stored: text_id={text_id}, vec_id={vec_id}")
    
    print(f"\nStore has {len(store.list_pairs())} text-embedding pairs")
    print()


# ============================================================================
# Example 3: Semantic Search
# ============================================================================

def example_semantic_search():
    """Perform semantic search using vector similarity."""
    print("=" * 60)
    print("Example 3: Semantic Search")
    print("=" * 60)
    
    # Create store and add documents
    store = ISStore("./search_store")
    
    docs = [
        "Python is a versatile programming language.",
        "Dogs are loyal and friendly animals.",
        "Machine learning uses statistical techniques.",
        "Cats are independent and curious pets.",
        "Deep learning is a subset of machine learning.",
    ]
    
    # Store with deterministic embeddings for reproducibility
    for i, doc in enumerate(docs):
        np.random.seed(hash(doc) % 2**32)
        embedding = np.random.randn(1536).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        store.append_text_with_embedding(doc, embedding, quantize=False)
    
    # Search query
    query = "artificial intelligence and computing"
    np.random.seed(hash(query) % 2**32)
    query_vec = np.random.randn(1536).astype(np.float32)
    query_vec = query_vec / np.linalg.norm(query_vec)
    
    print(f"Query: '{query}'")
    print("\nTop 3 results:")
    
    results = store.search_by_vector(query_vec, top_k=3)
    for rank, (score, text_id, vec_id) in enumerate(results, 1):
        text = store.read_text(text_id)
        print(f"  {rank}. Score: {score:.4f} | {text}")
    
    print()


# ============================================================================
# Example 4: Multi-Store Search
# ============================================================================

def example_multi_store():
    """Search across multiple stores simultaneously."""
    print("=" * 60)
    print("Example 4: Multi-Store Search")
    print("=" * 60)
    
    # Create two separate stores
    wiki_store = ISStore("./multi_wiki")
    chat_store = ISStore("./multi_chat")
    
    # Add documents to wiki store
    wiki_docs = [
        "Albert Einstein developed the theory of relativity.",
        "The speed of light is approximately 299,792 km/s.",
    ]
    
    # Add documents to chat store
    chat_docs = [
        "User asked about physics yesterday.",
        "Previous conversation covered quantum mechanics.",
    ]
    
    # Populate stores
    for doc in wiki_docs:
        np.random.seed(hash(doc) % 2**32)
        emb = np.random.randn(1536).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        wiki_store.append_text_with_embedding(doc, emb)
    
    for doc in chat_docs:
        np.random.seed(hash(doc) % 2**32)
        emb = np.random.randn(1536).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        chat_store.append_text_with_embedding(doc, emb)
    
    # Create multi-store
    multi = MultiStore([wiki_store, chat_store])
    
    # Search across both stores
    query = "physics"
    np.random.seed(hash(query) % 2**32)
    q_vec = np.random.randn(1536).astype(np.float32)
    q_vec = q_vec / np.linalg.norm(q_vec)
    
    print(f"Searching both stores for: '{query}'")
    results = multi.search(q_vec, per_shard_k=2, final_k=3)
    
    print("\nResults:")
    for score, text_id, vec_id, store_idx in results:
        source = ["wiki", "chat"][store_idx]
        text = multi.stores[store_idx].read_text(text_id)
        print(f"  [{source}] Score: {score:.4f} | {text}")
    
    print()


# ============================================================================
# Example 5: OpenAI Embeddings (requires API key)
# ============================================================================

def example_openai_embeddings():
    """Use OpenAI embeddings for real semantic search."""
    print("=" * 60)
    print("Example 5: OpenAI Embeddings")
    print("=" * 60)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set. Skipping this example.")
        print("   Set it with: export OPENAI_API_KEY='sk-...'")
        print()
        return
    
    try:
        client = get_client()
        store = ISStore("./openai_store")
        
        docs = [
            "Quantum computing uses quantum mechanics principles.",
            "Classical computers use binary bits for computation.",
            "Quantum bits (qubits) can exist in superposition.",
        ]
        
        print("Embedding and storing documents with OpenAI:")
        for doc in docs:
            embedding = embed_text_1536(client, doc)
            text_id, vec_id = store.append_text_with_embedding(doc, embedding, quantize=True)
            print(f"  ✓ Stored: {doc[:50]}...")
        
        # Search
        query = "How do quantum computers work?"
        print(f"\nQuery: '{query}'")
        query_vec = embed_text_1536(client, query)
        
        results = store.search_by_vector(query_vec, top_k=2)
        print("\nResults:")
        for rank, (score, tid, vid) in enumerate(results, 1):
            text = store.read_text(tid)
            print(f"  {rank}. Score: {score:.4f} | {text}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print()


# ============================================================================
# Example 6: Store Statistics
# ============================================================================

def example_store_stats():
    """Get statistics about a store."""
    print("=" * 60)
    print("Example 6: Store Statistics")
    print("=" * 60)
    
    store = ISStore("./stats_store")
    
    # Add some data
    for i in range(10):
        text = f"Document number {i} with some content."
        emb = np.random.randn(1536).astype(np.float32)
        store.append_text_with_embedding(text, emb, quantize=(i % 2 == 0))
    
    # Get stats
    pairs = store.list_pairs()
    text_segs = store.list_segments(data_type=0)  # DT_TEXT
    vec_segs = store.list_segments(data_type=1)   # DT_VEC_F32
    vec_i8_segs = store.list_segments(data_type=2)  # DT_VEC_I8
    
    print(f"Total text-embedding pairs: {len(pairs)}")
    print(f"Text segments: {len(text_segs)}")
    print(f"Float32 vector segments: {len(vec_segs)}")
    print(f"Int8 vector segments: {len(vec_i8_segs)}")
    
    # Calculate storage size
    import os
    total_size = 0
    for seg_id in store._glob_segments():
        path = store._segment_path(seg_id)
        total_size += os.path.getsize(path)
    
    print(f"Total storage size: {total_size:,} bytes ({total_size/1024:.1f} KB)")
    print()


# ============================================================================
# Example 7: Batch Operations
# ============================================================================

def example_batch_operations():
    """Efficiently process many documents."""
    print("=" * 60)
    print("Example 7: Batch Operations")
    print("=" * 60)
    
    store = ISStore("./batch_store")
    
    # Generate batch of documents
    num_docs = 100
    print(f"Creating {num_docs} documents...")
    
    import time
    start = time.time()
    
    for i in range(num_docs):
        text = f"This is document {i} with unique content about topic {i % 10}."
        emb = np.random.randn(1536).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        store.append_text_with_embedding(text, emb, quantize=True)
    
    elapsed = time.time() - start
    print(f"✓ Stored {num_docs} documents in {elapsed:.2f} seconds")
    print(f"  Rate: {num_docs/elapsed:.1f} docs/sec")
    
    # Quick search
    query_vec = np.random.randn(1536).astype(np.float32)
    query_vec = query_vec / np.linalg.norm(query_vec)
    
    start = time.time()
    results = store.search_by_vector(query_vec, top_k=10)
    search_time = time.time() - start
    
    print(f"✓ Searched {num_docs} documents in {search_time*1000:.1f} ms")
    print(f"  Found {len(results)} results")
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("ContextTape Quickstart Examples")
    print("=" * 60 + "\n")
    
    examples = [
        example_basic_store,
        example_text_with_embeddings,
        example_semantic_search,
        example_multi_store,
        example_openai_embeddings,
        example_store_stats,
        example_batch_operations,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"Error in {example.__name__}: {e}")
            print()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
