#!/usr/bin/env python3
"""
Test ContextTape with Wikipedia
================================

Quick test to ingest Wikipedia pages and search them.
"""

from contexttape import TSStore, embed_text_1536, get_client
from contexttape.ingest_wiki import fetch_wiki_page
import numpy as np

print("="*60)
print("Wikipedia Test")
print("="*60)
print()

# Topics to fetch
topics = [
    "Python_(programming_language)",
    "Machine_learning",
    "Artificial_intelligence",
    "Neural_network",
]

print(f"Fetching {len(topics)} Wikipedia pages...")
print()

# Create store
store = TSStore("data/wiki_test")

# Get OpenAI client (or use fake embeddings)
try:
    client = get_client()
    use_openai = True
    print("✓ Using OpenAI embeddings")
except:
    use_openai = False
    print("✓ Using fake embeddings (no OPENAI_API_KEY)")
print()

# Fetch and ingest
for i, topic in enumerate(topics, 1):
    print(f"[{i}/{len(topics)}] {topic.replace('_', ' ')}...")
    
    # Fetch Wikipedia page
    text = fetch_wiki_page(topic, verbose=False)
    
    # Create embedding
    if use_openai:
        vec = embed_text_1536(client, text[:8000])  # Limit to 8K chars
    else:
        # Fake embedding for testing
        vec = np.random.randn(1536).astype(np.float32)
    
    # Store it
    text_id = store.append_text(text)
    vec_id = store.append_vector_i8(vec, prev_text_id=text_id)
    
    print(f"  ✓ Stored: text_id={text_id}, vec_id={vec_id}")

print()
print("="*60)
print("Search Test")
print("="*60)
print()

# Search query
query = "What is machine learning?"
print(f"Query: {query}")
print()

if use_openai:
    query_vec = embed_text_1536(client, query)
else:
    query_vec = np.random.randn(1536).astype(np.float32)

# Search
results = store.search_by_vector(query_vec, top_k=3)

print(f"Top {len(results)} results:")
print()
for i, (score, text_id, vec_id) in enumerate(results, 1):
    text = store.read_text(text_id)
    preview = text[:200].replace('\n', ' ')
    print(f"{i}. Score: {score:.4f}")
    print(f"   {preview}...")
    print()

print("="*60)
print("✅ Wikipedia test complete!")
print("="*60)
print()
print(f"Data stored in: data/wiki_test/")
print(f"To clean up: rm -rf data/wiki_test/")
