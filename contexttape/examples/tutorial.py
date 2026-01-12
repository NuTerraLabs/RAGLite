"""
ContextTape Usage Tutorial
===========================

A step-by-step guide to using ContextTape for various use cases.

NOTE: These tutorials create temporary store directories (tutorial_*).
      These are runtime-generated user data, NOT part of the package.
      Clean up afterward: bash cleanup_stores.sh
"""

# ==============================================================================
# Tutorial 1: Getting Started
# ==============================================================================

def tutorial_1_getting_started():
    """
    Learn the basics: creating a store, adding documents, and retrieving them.
    """
    print("\n" + "="*70)
    print("Tutorial 1: Getting Started")
    print("="*70)
    
    from contexttape import ISStore
    import numpy as np
    
    # Step 1: Create a store
    print("\n1. Creating a store...")
    store = ISStore("./tutorial_store")
    print("   ✓ Store created at ./tutorial_store")
    
    # Step 2: Add a simple text document
    print("\n2. Adding a text document...")
    text_id = store.append_text("My first document in ContextTape!")
    print(f"   ✓ Document stored with ID: {text_id}")
    
    # Step 3: Read it back
    print("\n3. Reading the document back...")
    retrieved = store.read_text(text_id)
    print(f"   ✓ Retrieved: '{retrieved}'")
    
    # Step 4: Add document with embedding
    print("\n4. Adding a document with an embedding...")
    text = "Machine learning is transforming how we work with data."
    
    # Create a simple embedding (in production, use OpenAI or another model)
    embedding = np.random.randn(1536).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)  # Normalize
    
    text_id, vec_id = store.append_text_with_embedding(text, embedding)
    print(f"   ✓ Stored: text_id={text_id}, vector_id={vec_id}")
    
    # Step 5: Search
    print("\n5. Searching for similar documents...")
    query_vec = np.random.randn(1536).astype(np.float32)
    query_vec = query_vec / np.linalg.norm(query_vec)
    
    results = store.search_by_vector(query_vec, top_k=1)
    for score, tid, vid in results:
        found_text = store.read_text(tid)
        print(f"   ✓ Found (score={score:.4f}): '{found_text[:50]}...'")
    
    print("\n✓ Tutorial 1 complete!")


# ==============================================================================
# Tutorial 2: Working with Real Embeddings
# ==============================================================================

def tutorial_2_real_embeddings():
    """
    Use OpenAI embeddings for semantic search.
    Requires: OPENAI_API_KEY environment variable
    """
    print("\n" + "="*70)
    print("Tutorial 2: Working with Real Embeddings")
    print("="*70)
    
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  This tutorial requires OPENAI_API_KEY")
        print("   Set it with: export OPENAI_API_KEY='sk-...'")
        print("   Skipping...")
        return
    
    from contexttape import ISStore, get_client, embed_text_1536
    
    # Get OpenAI client
    print("\n1. Initializing OpenAI client...")
    client = get_client()
    print("   ✓ Client ready")
    
    # Create store
    store = ISStore("./tutorial_openai")
    
    # Add documents with real embeddings
    print("\n2. Adding documents with OpenAI embeddings...")
    documents = [
        "Python is a popular programming language for data science.",
        "Neural networks are the foundation of deep learning.",
        "The Renaissance was a period of cultural rebirth in Europe.",
        "Quantum computing uses quantum mechanical phenomena.",
    ]
    
    for i, doc in enumerate(documents):
        print(f"   Embedding document {i+1}/{len(documents)}...")
        embedding = embed_text_1536(client, doc)
        store.append_text_with_embedding(doc, embedding, quantize=True)
    print("   ✓ All documents stored")
    
    # Search with semantic query
    print("\n3. Performing semantic search...")
    query = "artificial intelligence and machine learning"
    print(f"   Query: '{query}'")
    
    query_embedding = embed_text_1536(client, query)
    results = store.search_by_vector(query_embedding, top_k=3)
    
    print("\n   Top results:")
    for rank, (score, tid, vid) in enumerate(results, 1):
        text = store.read_text(tid)
        print(f"   {rank}. [{score:.4f}] {text}")
    
    print("\n✓ Tutorial 2 complete!")


# ==============================================================================
# Tutorial 3: Building a Knowledge Base
# ==============================================================================

def tutorial_3_knowledge_base():
    """
    Build a searchable knowledge base from multiple documents.
    """
    print("\n" + "="*70)
    print("Tutorial 3: Building a Knowledge Base")
    print("="*70)
    
    from contexttape import ISStore
    import numpy as np
    
    # Sample knowledge base
    kb_documents = {
        "Python Basics": "Python is an interpreted, high-level programming language known for its simplicity.",
        "Machine Learning": "ML is a subset of AI that enables systems to learn from data without explicit programming.",
        "Deep Learning": "Deep learning uses neural networks with multiple layers to learn hierarchical representations.",
        "Data Science": "Data science combines statistics, programming, and domain knowledge to extract insights.",
        "Neural Networks": "Neural networks are computing systems inspired by biological neural networks.",
        "Natural Language Processing": "NLP is a field of AI focused on interaction between computers and human language.",
        "Computer Vision": "Computer vision enables computers to derive meaningful information from visual inputs.",
        "Reinforcement Learning": "RL is learning what actions to take to maximize reward in an environment.",
    }
    
    print(f"\n1. Creating knowledge base with {len(kb_documents)} articles...")
    store = ISStore("./tutorial_kb")
    
    # Ingest all documents
    for title, content in kb_documents.items():
        # Create deterministic embedding for demo
        np.random.seed(hash(content) % 2**32)
        embedding = np.random.randn(1536).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        
        # Store with title as prefix
        full_text = f"{title}: {content}"
        store.append_text_with_embedding(full_text, embedding, quantize=True)
    
    print(f"   ✓ {len(kb_documents)} articles indexed")
    
    # Query the knowledge base
    print("\n2. Querying the knowledge base...")
    queries = [
        "What is AI and ML?",
        "Tell me about neural networks",
        "How does computer vision work?"
    ]
    
    for query in queries:
        print(f"\n   Q: {query}")
        
        # Create query embedding
        np.random.seed(hash(query) % 2**32)
        q_vec = np.random.randn(1536).astype(np.float32)
        q_vec = q_vec / np.linalg.norm(q_vec)
        
        # Search
        results = store.search_by_vector(q_vec, top_k=2)
        
        for rank, (score, tid, vid) in enumerate(results, 1):
            text = store.read_text(tid)
            title = text.split(":")[0]
            print(f"      {rank}. {title} (relevance: {score:.3f})")
    
    print("\n✓ Tutorial 3 complete!")


# ==============================================================================
# Tutorial 4: Multi-Source RAG
# ==============================================================================

def tutorial_4_multi_source_rag():
    """
    Search across multiple knowledge sources (wiki + chat history).
    """
    print("\n" + "="*70)
    print("Tutorial 4: Multi-Source RAG")
    print("="*70)
    
    from contexttape import ISStore, MultiStore
    import numpy as np
    
    # Create separate stores for different sources
    print("\n1. Creating multiple knowledge sources...")
    
    wiki_store = ISStore("./tutorial_wiki")
    chat_store = ISStore("./tutorial_chat")
    docs_store = ISStore("./tutorial_docs")
    
    # Populate wiki store
    wiki_docs = [
        "Albert Einstein developed the theory of relativity.",
        "Python was created by Guido van Rossum in 1991.",
        "The human brain contains approximately 86 billion neurons.",
    ]
    
    # Populate chat history
    chat_history = [
        "User: What's a good programming language for beginners?",
        "Assistant: Python is excellent for beginners due to its simple syntax.",
        "User: Tell me about machine learning",
    ]
    
    # Populate documentation
    docs_content = [
        "API Documentation: Use the ISStore class to create a new segment store.",
        "Configuration: Set OPENAI_API_KEY environment variable for embeddings.",
    ]
    
    # Ingest into respective stores
    for text in wiki_docs:
        np.random.seed(hash(text) % 2**32)
        emb = np.random.randn(1536).astype(np.float32)
        wiki_store.append_text_with_embedding(text, emb / np.linalg.norm(emb))
    
    for text in chat_history:
        np.random.seed(hash(text) % 2**32)
        emb = np.random.randn(1536).astype(np.float32)
        chat_store.append_text_with_embedding(text, emb / np.linalg.norm(emb))
    
    for text in docs_content:
        np.random.seed(hash(text) % 2**32)
        emb = np.random.randn(1536).astype(np.float32)
        docs_store.append_text_with_embedding(text, emb / np.linalg.norm(emb))
    
    print(f"   ✓ Wiki: {len(wiki_docs)} articles")
    print(f"   ✓ Chat: {len(chat_history)} messages")
    print(f"   ✓ Docs: {len(docs_content)} pages")
    
    # Create multi-store for unified search
    print("\n2. Creating unified search across all sources...")
    multi = MultiStore([wiki_store, chat_store, docs_store])
    print("   ✓ Multi-store created")
    
    # Search across all sources
    print("\n3. Searching across all sources...")
    query = "Python programming"
    np.random.seed(hash(query) % 2**32)
    q_vec = np.random.randn(1536).astype(np.float32)
    q_vec = q_vec / np.linalg.norm(q_vec)
    
    results = multi.search(q_vec, per_shard_k=2, final_k=4)
    
    source_names = ["wiki", "chat", "docs"]
    print(f"\n   Results for: '{query}'")
    for store_path, score, tid, vid in results:
        # Determine source
        if "wiki" in store_path:
            source = "wiki"
            text = wiki_store.read_text(tid)
        elif "chat" in store_path:
            source = "chat"
            text = chat_store.read_text(tid)
        else:
            source = "docs"
            text = docs_store.read_text(tid)
        
        print(f"   [{source:5s}] {score:.3f} | {text[:60]}...")
    
    print("\n✓ Tutorial 4 complete!")


# ==============================================================================
# Tutorial 5: Production Deployment
# ==============================================================================

def tutorial_5_production():
    """
    Best practices for production deployment.
    """
    print("\n" + "="*70)
    print("Tutorial 5: Production Deployment Patterns")
    print("="*70)
    
    print("""
Production Checklist:
    
1. ✓ Use quantization for 4x space savings
   store.append_text_with_embedding(text, emb, quantize=True)

2. ✓ Batch operations for efficiency
   store.append_batch(texts, embeddings, quantize=True)

3. ✓ Monitor store statistics
   stats = store.stat()
   print(f"Pairs: {stats['pairs']}, Next ID: {stats['next_id']}")

4. ✓ Regular compaction to remove orphaned segments
   compact_stats = store.compact()

5. ✓ Export for backup
   export_data = store.export_to_dict(include_vectors=False)
   with open('backup.json', 'w') as f:
       json.dump(export_data, f)

6. ✓ Use stride for faster (approximate) search
   results = store.search_by_vector(q, top_k=10, stride=2)

7. ✓ Implement hybrid search for better quality
   from contexttape import combined_search
   results = combined_search(query, query_vec, wiki_store, 
                            chat_store, top_k=5)

8. ✓ Set up monitoring and logging
   - Track query latency
   - Monitor storage size
   - Log failed embeddings
   - Alert on errors

9. ✓ Consider multi-store architecture
   - Separate stores for different data types
   - Hot/cold data separation
   - Per-tenant stores for multi-tenancy

10. ✓ Implement graceful error handling
    try:
        results = store.search_by_vector(q, top_k=5)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return fallback_results()
""")
    
    print("\n✓ Tutorial 5 complete!")


# ==============================================================================
# Main
# ==============================================================================

def main():
    """Run all tutorials."""
    print("\n" + "="*70)
    print("ContextTape Complete Tutorial")
    print("="*70)
    
    tutorials = [
        tutorial_1_getting_started,
        tutorial_2_real_embeddings,
        tutorial_3_knowledge_base,
        tutorial_4_multi_source_rag,
        tutorial_5_production,
    ]
    
    for tutorial in tutorials:
        try:
            tutorial()
        except Exception as e:
            print(f"\n❌ Error in {tutorial.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("All Tutorials Complete!")
    print("="*70)
    print("\nNext steps:")
    print("  • Check out examples/advanced_usage.py for more patterns")
    print("  • Read the full documentation at docs/")
    print("  • Try integrating with your own data")
    print("  • Star the repo if you find it useful!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
