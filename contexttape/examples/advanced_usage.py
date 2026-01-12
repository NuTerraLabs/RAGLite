"""
Advanced ContextTape Usage Examples
====================================

Advanced patterns and integrations for production use.

NOTE: These examples create temporary store directories (*_store, *_ts).
      These are runtime-generated user data, NOT part of the package.
      Clean up afterward: bash cleanup_stores.sh
"""

import numpy as np
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
from contexttape import (
    ISStore, 
    MultiStore, 
    ISHeader,
    DT_JSON,
    DT_BLOB,
    ContextTapeClient,
)


# ============================================================================
# Example 1: Custom Metadata with JSON Segments
# ============================================================================

def example_json_metadata():
    """Store structured metadata alongside text."""
    print("=" * 60)
    print("Example 1: JSON Metadata Storage")
    print("=" * 60)
    
    store = ISStore("./metadata_store")
    
    # Store document with metadata
    documents = [
        {
            "text": "Introduction to machine learning fundamentals.",
            "metadata": {
                "author": "Dr. Smith",
                "date": "2024-01-15",
                "category": "AI",
                "tags": ["ml", "tutorial", "beginner"],
                "version": 1,
            }
        },
        {
            "text": "Advanced neural network architectures explained.",
            "metadata": {
                "author": "Dr. Johnson",
                "date": "2024-02-20",
                "category": "Deep Learning",
                "tags": ["neural-networks", "advanced"],
                "version": 2,
            }
        },
    ]
    
    print("Storing documents with metadata:")
    for doc in documents:
        # Store text
        text_id = store.append_text(doc["text"])
        
        # Store metadata as JSON
        metadata_json = json.dumps(doc["metadata"]).encode("utf-8")
        meta_id = store.append_json(metadata_json)
        
        # Link them
        store.link_text_to_vec(text_id, meta_id)
        
        print(f"  ✓ Document {text_id}: {doc['text'][:50]}...")
        print(f"    Metadata {meta_id}: {doc['metadata']['author']}")
    
    # Retrieve with metadata
    print("\nRetrieving documents with metadata:")
    for text_id, meta_id in store.list_pairs():
        text = store.read_text(text_id)
        metadata_bytes = store.read_json(meta_id)
        metadata = json.loads(metadata_bytes.decode("utf-8"))
        
        print(f"\n  Document {text_id}:")
        print(f"    Text: {text}")
        print(f"    Author: {metadata['author']}")
        print(f"    Tags: {', '.join(metadata['tags'])}")
    
    print()


# ============================================================================
# Example 2: Binary Blob Storage
# ============================================================================

def example_blob_storage():
    """Store arbitrary binary data (images, audio, etc.)."""
    print("=" * 60)
    print("Example 2: Binary Blob Storage")
    print("=" * 60)
    
    store = ISStore("./blob_store")
    
    # Simulate storing an image
    fake_image_data = b"\x89PNG\r\n\x1a\n" + os.urandom(1024)  # Fake PNG header + random data
    
    # Store blob
    blob_id = store.append_blob(fake_image_data)
    print(f"Stored blob (simulated image): segment {blob_id}, size: {len(fake_image_data)} bytes")
    
    # Store metadata about the blob
    metadata = {
        "type": "image/png",
        "width": 640,
        "height": 480,
        "description": "Sample image for ML training"
    }
    text_id = store.append_text(json.dumps(metadata))
    
    # Link metadata to blob
    store.link_text_to_vec(text_id, blob_id)
    
    # Retrieve
    retrieved_blob = store.read_blob(blob_id)
    retrieved_meta = json.loads(store.read_text(text_id))
    
    print(f"Retrieved blob: {len(retrieved_blob)} bytes")
    print(f"Metadata: {retrieved_meta}")
    print(f"Blob matches original: {retrieved_blob == fake_image_data}")
    print()


# ============================================================================
# Example 3: Filtered Search
# ============================================================================

def example_filtered_search():
    """Search with filters and post-processing."""
    print("=" * 60)
    print("Example 3: Filtered Search")
    print("=" * 60)
    
    store = ISStore("./filtered_store")
    
    # Add documents with categories
    docs = [
        ("Python basics for beginners", "programming"),
        ("Machine learning fundamentals", "ai"),
        ("Web development with Flask", "programming"),
        ("Deep learning techniques", "ai"),
        ("Database design principles", "programming"),
    ]
    
    # Store with embeddings
    doc_info = []
    for text, category in docs:
        np.random.seed(hash(text) % 2**32)
        emb = np.random.randn(1536).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        text_id, vec_id = store.append_text_with_embedding(text, emb)
        doc_info.append((text_id, category))
    
    # Search with category filter
    query = "learning algorithms"
    np.random.seed(hash(query) % 2**32)
    q_vec = np.random.randn(1536).astype(np.float32)
    q_vec = q_vec / np.linalg.norm(q_vec)
    
    # Get all results
    all_results = store.search_by_vector(q_vec, top_k=10)
    
    # Filter for "ai" category only
    filter_category = "ai"
    filtered_results = []
    for score, tid, vid in all_results:
        # Find category for this document
        for doc_tid, cat in doc_info:
            if doc_tid == tid and cat == filter_category:
                filtered_results.append((score, tid, vid))
                break
    
    print(f"Query: '{query}'")
    print(f"Filter: category='{filter_category}'")
    print(f"\nFiltered results ({len(filtered_results)} found):")
    for score, tid, vid in filtered_results:
        text = store.read_text(tid)
        print(f"  Score: {score:.4f} | {text}")
    
    print()


# ============================================================================
# Example 4: Hierarchical Stores
# ============================================================================

def example_hierarchical_stores():
    """Organize data in hierarchical stores."""
    print("=" * 60)
    print("Example 4: Hierarchical Stores")
    print("=" * 60)
    
    # Create stores for different domains
    stores = {
        "science": ISStore("./hierarchy/science"),
        "history": ISStore("./hierarchy/history"),
        "technology": ISStore("./hierarchy/technology"),
    }
    
    # Add domain-specific content
    content = {
        "science": [
            "Photosynthesis converts light energy into chemical energy.",
            "DNA contains genetic information in all living organisms.",
        ],
        "history": [
            "The Roman Empire fell in 476 CE.",
            "The Renaissance began in Italy in the 14th century.",
        ],
        "technology": [
            "Artificial intelligence mimics human cognitive functions.",
            "Blockchain provides decentralized data storage.",
        ],
    }
    
    # Populate stores
    for domain, texts in content.items():
        print(f"\nPopulating {domain} store:")
        for text in texts:
            np.random.seed(hash(text) % 2**32)
            emb = np.random.randn(1536).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            tid, vid = stores[domain].append_text_with_embedding(text, emb)
            print(f"  ✓ {text[:50]}...")
    
    # Search specific domain
    query = "genetic information"
    np.random.seed(hash(query) % 2**32)
    q_vec = np.random.randn(1536).astype(np.float32)
    q_vec = q_vec / np.linalg.norm(q_vec)
    
    print(f"\n\nSearching science domain for: '{query}'")
    results = stores["science"].search_by_vector(q_vec, top_k=2)
    for score, tid, vid in results:
        text = stores["science"].read_text(tid)
        print(f"  Score: {score:.4f} | {text}")
    
    # Search all domains
    print(f"\nSearching ALL domains:")
    multi = MultiStore(list(stores.values()))
    all_results = multi.search(q_vec, per_shard_k=2, final_k=3)
    for score, tid, vid, store_idx in all_results:
        domain = list(stores.keys())[store_idx]
        text = list(stores.values())[store_idx].read_text(tid)
        print(f"  [{domain}] Score: {score:.4f} | {text}")
    
    print()


# ============================================================================
# Example 5: High-Level Client API
# ============================================================================

def example_client_api():
    """Use the high-level ContextTapeClient."""
    print("=" * 60)
    print("Example 5: High-Level Client API")
    print("=" * 60)
    
    # Create client (abstracts store management)
    client = ContextTapeClient("./client_store")
    
    # Ingest documents (client handles embeddings if API key is set)
    documents = [
        "Quantum entanglement is a physical phenomenon.",
        "Machine learning models require training data.",
        "Python is an interpreted programming language.",
    ]
    
    print("Ingesting documents:")
    for doc in documents:
        # For demo, we'll use random embeddings
        emb = np.random.randn(1536).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        client.ingest(doc, embedding=emb)
        print(f"  ✓ {doc[:50]}...")
    
    # Search using client
    query = "quantum physics"
    q_emb = np.random.randn(1536).astype(np.float32)
    q_emb = q_emb / np.linalg.norm(q_emb)
    
    print(f"\nSearching for: '{query}'")
    results = client.search(query, query_embedding=q_emb, top_k=2)
    
    for result in results:
        print(f"  Score: {result['score']:.4f}")
        print(f"  Text: {result['text']}")
        print()


# ============================================================================
# Example 6: Export and Backup
# ============================================================================

def example_export_backup():
    """Export store data for backup or transfer."""
    print("=" * 60)
    print("Example 6: Export and Backup")
    print("=" * 60)
    
    # Create and populate store
    store = ISStore("./export_store")
    docs = [
        "Document one with important data.",
        "Document two with more information.",
        "Document three for completeness.",
    ]
    
    for doc in docs:
        emb = np.random.randn(1536).astype(np.float32)
        store.append_text_with_embedding(doc, emb, quantize=True)
    
    # Export store information
    export_data = {
        "version": "1.0",
        "total_pairs": len(store.list_pairs()),
        "documents": []
    }
    
    for text_id, vec_id in store.list_pairs():
        text = store.read_text(text_id)
        vector = store.read_vector(vec_id)
        
        export_data["documents"].append({
            "text_id": text_id,
            "vec_id": vec_id,
            "text": text,
            "vector_dim": len(vector),
            "vector_norm": float(np.linalg.norm(vector)),
        })
    
    # Save export
    export_path = "./export_store/export.json"
    with open(export_path, "w") as f:
        json.dump(export_data, f, indent=2)
    
    print(f"Exported {len(export_data['documents'])} documents to {export_path}")
    print(f"Export size: {os.path.getsize(export_path)} bytes")
    
    # Show export contents
    print("\nExport summary:")
    print(f"  Version: {export_data['version']}")
    print(f"  Total pairs: {export_data['total_pairs']}")
    print(f"  First document: {export_data['documents'][0]['text'][:50]}...")
    print()


# ============================================================================
# Example 7: Performance Monitoring
# ============================================================================

def example_performance_monitoring():
    """Monitor store performance metrics."""
    print("=" * 60)
    print("Example 7: Performance Monitoring")
    print("=" * 60)
    
    import time
    
    store = ISStore("./perf_store")
    
    # Benchmark ingestion
    num_docs = 50
    print(f"Benchmarking ingestion of {num_docs} documents...")
    
    start = time.time()
    for i in range(num_docs):
        text = f"Performance test document {i} with content."
        emb = np.random.randn(1536).astype(np.float32)
        store.append_text_with_embedding(text, emb, quantize=True)
    ingest_time = time.time() - start
    
    print(f"  ✓ Ingestion: {ingest_time:.2f}s ({num_docs/ingest_time:.1f} docs/sec)")
    
    # Benchmark search
    queries = 10
    print(f"\nBenchmarking {queries} searches...")
    
    search_times = []
    for _ in range(queries):
        q_vec = np.random.randn(1536).astype(np.float32)
        q_vec = q_vec / np.linalg.norm(q_vec)
        
        start = time.time()
        results = store.search_by_vector(q_vec, top_k=5)
        search_times.append(time.time() - start)
    
    avg_search = np.mean(search_times) * 1000  # Convert to ms
    p95_search = np.percentile(search_times, 95) * 1000
    
    print(f"  ✓ Search avg: {avg_search:.2f}ms")
    print(f"  ✓ Search p95: {p95_search:.2f}ms")
    
    # Storage efficiency
    total_bytes = sum(
        os.path.getsize(store._segment_path(sid))
        for sid in store._glob_segments()
    )
    bytes_per_doc = total_bytes / num_docs
    
    print(f"\nStorage efficiency:")
    print(f"  Total size: {total_bytes:,} bytes ({total_bytes/1024:.1f} KB)")
    print(f"  Per document: {bytes_per_doc:.0f} bytes")
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all advanced examples."""
    print("\n" + "=" * 60)
    print("Advanced ContextTape Examples")
    print("=" * 60 + "\n")
    
    examples = [
        example_json_metadata,
        example_blob_storage,
        example_filtered_search,
        example_hierarchical_stores,
        example_client_api,
        example_export_backup,
        example_performance_monitoring,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"Error in {example.__name__}: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 60)
    print("All advanced examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
