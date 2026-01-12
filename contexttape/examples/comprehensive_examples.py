#!/usr/bin/env python3
"""
ContextTape Examples
====================

This file contains comprehensive examples demonstrating all major features
of ContextTape. Run each section independently or all together.

Requirements:
    pip install contexttape
    export OPENAI_API_KEY="sk-..."  # For embedding generation

Contents:
    1. Basic Usage - Store creation, ingestion, search
    2. Quantization - 4x smaller storage with int8
    3. Multi-Store - Searching across multiple stores
    4. Batch Processing - Efficient bulk operations
    5. Custom Embeddings - Using your own embedding model
    6. Streaming Search - Large corpus handling
    7. Export/Import - Backup and migration
    8. FastAPI Server - REST API deployment
    9. LangChain Integration - RAG chains
    10. Advanced Search - Hybrid scoring
"""

import os
import sys
import tempfile
import shutil
from typing import List
import numpy as np


# =============================================================================
# 1. BASIC USAGE
# =============================================================================

def example_basic_usage():
    """
    Basic usage: create a store, add documents, and search.
    """
    print("\n" + "="*60)
    print("1. BASIC USAGE")
    print("="*60)
    
    from contexttape import ISStore, get_client, embed_text_1536
    
    # Create a temporary store
    store_path = tempfile.mkdtemp(prefix="contexttape_example_")
    store = ISStore(store_path)
    
    print(f"Created store at: {store_path}")
    
    # Get OpenAI client for embeddings
    try:
        client = get_client()
    except RuntimeError:
        print("⚠️  Set OPENAI_API_KEY to run embedding examples")
        print("    Using random vectors for demo...")
        client = None
    
    # Sample documents
    documents = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Neural networks are computing systems inspired by biological neural networks in animal brains.",
        "Deep learning uses multiple layers of neural networks to progressively extract features.",
        "Natural language processing enables computers to understand and generate human language.",
        "Computer vision allows machines to interpret and make decisions based on visual data.",
    ]
    
    # Ingest documents
    print("\nIngesting documents...")
    for i, doc in enumerate(documents):
        if client:
            embedding = embed_text_1536(client, doc)
        else:
            # Demo fallback: deterministic random vector
            np.random.seed(hash(doc) % 2**32)
            embedding = np.random.randn(1536).astype(np.float32)
        
        text_id, vec_id = store.append_text_with_embedding(doc, embedding)
        print(f"  [{i+1}] text_id={text_id}, vec_id={vec_id}")
    
    # Search
    query = "How do neural networks work?"
    print(f"\nSearching for: '{query}'")
    
    if client:
        query_vec = embed_text_1536(client, query)
    else:
        np.random.seed(hash(query) % 2**32)
        query_vec = np.random.randn(1536).astype(np.float32)
    
    results = store.search_by_vector(query_vec, top_k=3)
    
    print("\nTop 3 results:")
    for rank, (score, tid, eid) in enumerate(results, 1):
        text = store.read_text(tid)
        print(f"  [{rank}] score={score:.4f}")
        print(f"      {text[:80]}...")
    
    # Stats
    print(f"\nStore stats: {store.stat()}")
    
    # Cleanup
    shutil.rmtree(store_path)
    print(f"\nCleaned up {store_path}")
    
    return True


# =============================================================================
# 2. QUANTIZATION
# =============================================================================

def example_quantization():
    """
    Demonstrate int8 quantization for 4x smaller storage.
    """
    print("\n" + "="*60)
    print("2. QUANTIZATION (4x smaller storage)")
    print("="*60)
    
    from contexttape import ISStore
    
    # Create two stores for comparison
    store_f32_path = tempfile.mkdtemp(prefix="contexttape_f32_")
    store_i8_path = tempfile.mkdtemp(prefix="contexttape_i8_")
    
    store_f32 = ISStore(store_f32_path)
    store_i8 = ISStore(store_i8_path)
    
    # Generate sample data
    n_docs = 100
    dim = 1536
    
    print(f"Adding {n_docs} documents with {dim}-dim vectors...")
    
    for i in range(n_docs):
        text = f"Document number {i} with some sample content for testing."
        vec = np.random.randn(dim).astype(np.float32)
        
        # Store as float32
        store_f32.append_text_with_embedding(text, vec, quantize=False)
        
        # Store as int8 (quantized)
        store_i8.append_text_with_embedding(text, vec, quantize=True)
    
    # Compare sizes
    def get_dir_size(path):
        total = 0
        for f in os.listdir(path):
            total += os.path.getsize(os.path.join(path, f))
        return total
    
    size_f32 = get_dir_size(store_f32_path)
    size_i8 = get_dir_size(store_i8_path)
    
    print(f"\nStorage comparison:")
    print(f"  Float32: {size_f32:,} bytes ({size_f32/1024:.1f} KB)")
    print(f"  Int8:    {size_i8:,} bytes ({size_i8/1024:.1f} KB)")
    print(f"  Ratio:   {size_f32/size_i8:.2f}x smaller with quantization")
    
    # Verify search accuracy is preserved
    query_vec = np.random.randn(dim).astype(np.float32)
    
    results_f32 = store_f32.search_by_vector(query_vec, top_k=5)
    results_i8 = store_i8.search_by_vector(query_vec, top_k=5)
    
    print(f"\nSearch accuracy comparison (same query):")
    print(f"  Float32 top scores: {[f'{s:.4f}' for s, _, _ in results_f32[:3]]}")
    print(f"  Int8 top scores:    {[f'{s:.4f}' for s, _, _ in results_i8[:3]]}")
    
    # Cleanup
    shutil.rmtree(store_f32_path)
    shutil.rmtree(store_i8_path)
    
    return True


# =============================================================================
# 3. MULTI-STORE SEARCH
# =============================================================================

def example_multi_store():
    """
    Search across multiple stores with late fusion.
    """
    print("\n" + "="*60)
    print("3. MULTI-STORE SEARCH")
    print("="*60)
    
    from contexttape import ISStore, MultiStore
    
    # Create separate stores for different content types
    wiki_path = tempfile.mkdtemp(prefix="contexttape_wiki_")
    chat_path = tempfile.mkdtemp(prefix="contexttape_chat_")
    docs_path = tempfile.mkdtemp(prefix="contexttape_docs_")
    
    wiki_store = ISStore(wiki_path)
    chat_store = ISStore(chat_path)
    docs_store = ISStore(docs_path)
    
    # Populate stores with different content
    print("Creating multi-store setup...")
    
    wiki_content = [
        "Wikipedia: Machine learning overview and history.",
        "Wikipedia: Neural network architectures explained.",
    ]
    
    chat_content = [
        "user: What is deep learning? assistant: Deep learning uses multiple layers...",
        "user: How do I train a model? assistant: Training involves feeding data...",
    ]
    
    docs_content = [
        "Technical docs: API reference for model training.",
        "Technical docs: Configuration guide for deployment.",
    ]
    
    dim = 1536
    for i, text in enumerate(wiki_content):
        vec = np.random.randn(dim).astype(np.float32)
        wiki_store.append_text_with_embedding(text, vec)
    
    for i, text in enumerate(chat_content):
        vec = np.random.randn(dim).astype(np.float32)
        chat_store.append_text_with_embedding(text, vec)
    
    for i, text in enumerate(docs_content):
        vec = np.random.randn(dim).astype(np.float32)
        docs_store.append_text_with_embedding(text, vec)
    
    # Create MultiStore
    multi = MultiStore([wiki_store, chat_store, docs_store])
    
    # Search across all stores
    query_vec = np.random.randn(dim).astype(np.float32)
    
    results = multi.search(
        query_vec,
        per_shard_k=3,  # Get top 3 from each store
        final_k=5,       # Return top 5 overall
    )
    
    print(f"\nMulti-store search results:")
    for i, (store_path, score, tid, eid) in enumerate(results, 1):
        store_name = os.path.basename(store_path)
        print(f"  [{i}] store={store_name}, score={score:.4f}, tid={tid}")
    
    # Cleanup
    shutil.rmtree(wiki_path)
    shutil.rmtree(chat_path)
    shutil.rmtree(docs_path)
    
    return True


# =============================================================================
# 4. HIGH-LEVEL CLIENT API
# =============================================================================

def example_client_api():
    """
    Use the high-level ContextTapeClient for simplified operations.
    """
    print("\n" + "="*60)
    print("4. HIGH-LEVEL CLIENT API")
    print("="*60)
    
    from contexttape import ContextTapeClient
    
    store_path = tempfile.mkdtemp(prefix="contexttape_client_")
    
    # Use custom embedding function (for demo without OpenAI)
    def mock_embed(text: str) -> np.ndarray:
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(1536).astype(np.float32)
    
    client = ContextTapeClient(
        store_path=store_path,
        embed_fn=mock_embed,
        quantize=True,
    )
    
    print(f"Created client with store at: {store_path}")
    
    # Single document ingestion
    result = client.ingest(
        "The quick brown fox jumps over the lazy dog.",
        metadata={"source": "example", "category": "animals"}
    )
    print(f"\nIngested single doc: text_id={result.text_id}, success={result.success}")
    
    # Batch ingestion
    docs = [
        "Python is a versatile programming language.",
        "JavaScript runs in web browsers.",
        "Rust focuses on memory safety.",
        "Go was designed at Google.",
    ]
    
    results = client.ingest_batch(
        docs,
        metadata_list=[{"lang": d.split()[0]} for d in docs],
        progress_callback=lambda cur, tot: print(f"  Progress: {cur}/{tot}")
    )
    
    success_count = sum(1 for r in results if r.success)
    print(f"\nBatch ingested {success_count}/{len(docs)} documents")
    
    # Search
    search_results = client.search("programming language", top_k=3)
    
    print(f"\nSearch results:")
    for r in search_results:
        print(f"  score={r.score:.4f}: {r.text[:50]}...")
        if r.metadata:
            print(f"    metadata: {r.metadata}")
    
    # Stats
    print(f"\nClient stats: {client.stats()}")
    print(f"Total documents: {len(client)}")
    
    # Cleanup
    shutil.rmtree(store_path)
    
    return True


# =============================================================================
# 5. STREAMING SEARCH
# =============================================================================

def example_streaming():
    """
    Stream search results for large stores.
    """
    print("\n" + "="*60)
    print("5. STREAMING SEARCH")
    print("="*60)
    
    from contexttape import ISStore, stream_search, iterate_store
    
    store_path = tempfile.mkdtemp(prefix="contexttape_stream_")
    store = ISStore(store_path)
    
    # Create larger dataset
    print("Creating dataset with 500 documents...")
    dim = 1536
    for i in range(500):
        text = f"Document {i}: Lorem ipsum dolor sit amet, consectetur adipiscing elit."
        vec = np.random.randn(dim).astype(np.float32)
        store.append_text_with_embedding(text, vec, quantize=True)
    
    # Streaming search - results come as they're found
    query_vec = np.random.randn(dim).astype(np.float32)
    
    print("\nStreaming search (stop after 3 results above threshold):")
    count = 0
    for score, tid, vid, text in stream_search(store, query_vec, min_score=0.0):
        print(f"  Found: score={score:.4f}, {text[:40]}...")
        count += 1
        if count >= 3:
            print("  (stopped early)")
            break
    
    # Iterate all documents
    print(f"\nIterating store (first 3 of {len(store.list_pairs())}):")
    for i, (tid, vid, text, vec) in enumerate(iterate_store(store)):
        print(f"  [{tid}] dim={vec.shape[0]}, text={text[:30]}...")
        if i >= 2:
            break
    
    # Cleanup
    shutil.rmtree(store_path)
    
    return True


# =============================================================================
# 6. EXPORT AND IMPORT
# =============================================================================

def example_export_import():
    """
    Export to JSONL and import back.
    """
    print("\n" + "="*60)
    print("6. EXPORT AND IMPORT")
    print("="*60)
    
    from contexttape import ISStore, export_to_jsonl, import_from_jsonl
    
    # Create and populate original store
    original_path = tempfile.mkdtemp(prefix="contexttape_original_")
    original = ISStore(original_path)
    
    print("Creating original store with 10 documents...")
    dim = 1536
    for i in range(10):
        text = f"Original document {i} with content."
        vec = np.random.randn(dim).astype(np.float32)
        original.append_text_with_embedding(text, vec)
    
    print(f"Original store: {original.stat()}")
    
    # Export to JSONL
    export_file = os.path.join(tempfile.gettempdir(), "contexttape_export.jsonl")
    count = export_to_jsonl(original, export_file)
    print(f"\nExported {count} records to {export_file}")
    
    # Show export file size
    export_size = os.path.getsize(export_file)
    print(f"Export file size: {export_size:,} bytes")
    
    # Import into new store
    imported_path = tempfile.mkdtemp(prefix="contexttape_imported_")
    imported = ISStore(imported_path)
    
    count = import_from_jsonl(imported, export_file, quantize=True)
    print(f"\nImported {count} records into new store")
    print(f"Imported store: {imported.stat()}")
    
    # Verify content matches
    orig_text = original.read_text(0)
    imp_text = imported.read_text(0)
    print(f"\nVerification - first doc matches: {orig_text == imp_text}")
    
    # Cleanup
    shutil.rmtree(original_path)
    shutil.rmtree(imported_path)
    os.remove(export_file)
    
    return True


# =============================================================================
# 7. FASTAPI SERVER (code example - don't run directly)
# =============================================================================

def example_fastapi_code():
    """
    Show how to create a FastAPI server.
    """
    print("\n" + "="*60)
    print("7. FASTAPI SERVER (code example)")
    print("="*60)
    
    code = '''
# server.py
from contexttape import create_fastapi_app

# Create the app
app = create_fastapi_app(
    store_path="my_store",
    title="My RAG API",
    quantize=True,
)

# Run with: uvicorn server:app --host 0.0.0.0 --port 8000

# Endpoints available:
#   POST /ingest         - Ingest single document
#   POST /ingest/batch   - Ingest multiple documents
#   POST /search         - Search for documents
#   GET  /stats          - Get store statistics
#   GET  /health         - Health check

# Example curl commands:

# Ingest a document:
# curl -X POST http://localhost:8000/ingest \\
#      -H "Content-Type: application/json" \\
#      -d '{"text": "Hello world", "metadata": {"source": "test"}}'

# Search:
# curl -X POST http://localhost:8000/search \\
#      -H "Content-Type: application/json" \\
#      -d '{"query": "hello", "top_k": 5}'

# Get stats:
# curl http://localhost:8000/stats
'''
    print(code)
    
    return True


# =============================================================================
# 8. LANGCHAIN INTEGRATION (code example)
# =============================================================================

def example_langchain_code():
    """
    Show LangChain integration code.
    """
    print("\n" + "="*60)
    print("8. LANGCHAIN INTEGRATION (code example)")
    print("="*60)
    
    code = '''
# langchain_example.py
from contexttape import ContextTapeRetriever
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Create retriever
retriever = ContextTapeRetriever(
    store_path="my_store",
    k=5,
    min_score=0.3,
)

# Use directly
docs = retriever.get_relevant_documents("What is machine learning?")
for doc in docs:
    print(f"Score: {doc.metadata['score']:.4f}")
    print(f"Content: {doc.page_content[:100]}...")
    print()

# Or use in a chain
llm = ChatOpenAI(model="gpt-4")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
)

answer = qa_chain.run("Explain neural networks in simple terms.")
print(answer)
'''
    print(code)
    
    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all examples."""
    print("="*60)
    print("CONTEXTTAPE EXAMPLES")
    print("="*60)
    
    examples = [
        ("Basic Usage", example_basic_usage),
        ("Quantization", example_quantization),
        ("Multi-Store", example_multi_store),
        ("Client API", example_client_api),
        ("Streaming", example_streaming),
        ("Export/Import", example_export_import),
        ("FastAPI Code", example_fastapi_code),
        ("LangChain Code", example_langchain_code),
    ]
    
    results = []
    for name, func in examples:
        try:
            success = func()
            results.append((name, "✅" if success else "❌"))
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            results.append((name, f"❌ {type(e).__name__}"))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, status in results:
        print(f"  {status} {name}")
    
    print("\n✅ All examples completed!")


if __name__ == "__main__":
    main()
