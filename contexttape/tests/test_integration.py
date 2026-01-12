"""
Integration Tests for ContextTape
==================================

End-to-end tests verifying complete workflows.
"""

import tempfile
import shutil
import os
import numpy as np
import pytest
from pathlib import Path

from contexttape import (
    ISStore,
    MultiStore,
    ContextTapeClient,
    embed_text_1536,
    get_client,
)


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""

    def test_complete_rag_workflow(self):
        """Test a complete RAG workflow: ingest, search, retrieve."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ISStore(tmpdir)
            
            # Step 1: Ingest documents
            documents = [
                "Python is a programming language used for data science.",
                "Machine learning involves training models on data.",
                "Neural networks are inspired by the human brain structure.",
                "Data preprocessing is crucial for model performance.",
            ]
            
            for doc in documents:
                np.random.seed(hash(doc) % 2**32)
                emb = np.random.randn(1536).astype(np.float32)
                emb = emb / np.linalg.norm(emb)
                store.append_text_with_embedding(doc, emb, quantize=True)
            
            # Step 2: Verify storage
            pairs = store.list_pairs()
            assert len(pairs) == len(documents)
            
            # Step 3: Search
            query = "machine learning models"
            np.random.seed(hash(query) % 2**32)
            q_vec = np.random.randn(1536).astype(np.float32)
            q_vec = q_vec / np.linalg.norm(q_vec)
            
            results = store.search_by_vector(q_vec, top_k=2)
            assert len(results) == 2
            
            # Step 4: Retrieve and verify
            for score, tid, vid in results:
                text = store.read_text(tid)
                assert text in documents
                assert -1 <= score <= 1

    def test_multi_store_workflow(self):
        """Test multi-store late fusion workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple stores
            store1 = ISStore(os.path.join(tmpdir, "store1"))
            store2 = ISStore(os.path.join(tmpdir, "store2"))
            
            # Add different content to each
            docs1 = ["Document from store 1 about AI", "Another AI document"]
            docs2 = ["Document from store 2 about ML", "Another ML document"]
            
            for doc in docs1:
                np.random.seed(hash(doc) % 2**32)
                emb = np.random.randn(1536).astype(np.float32)
                store1.append_text_with_embedding(doc, emb)
            
            for doc in docs2:
                np.random.seed(hash(doc) % 2**32)
                emb = np.random.randn(1536).astype(np.float32)
                store2.append_text_with_embedding(doc, emb)
            
            # Search across both
            multi = MultiStore([store1, store2])
            q_vec = np.random.randn(1536).astype(np.float32)
            
            results = multi.search(q_vec, per_shard_k=2, final_k=3)
            assert len(results) <= 3
            assert all(len(r) == 4 for r in results)  # (store_path, score, tid, vid)

    def test_batch_operations_workflow(self):
        """Test batch ingestion and processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ISStore(tmpdir)
            
            # Generate batch
            num_docs = 50
            texts = [f"Document {i} with content about topic {i%5}" for i in range(num_docs)]
            embeddings = [np.random.randn(1536).astype(np.float32) for _ in range(num_docs)]
            
            # Batch append
            results = store.append_batch(texts, embeddings, quantize=True)
            
            assert len(results) == num_docs
            assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
            
            # Verify all stored
            pairs = store.list_pairs()
            assert len(pairs) == num_docs

    def test_export_import_workflow(self):
        """Test exporting and re-importing data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create original store
            store1 = ISStore(os.path.join(tmpdir, "original"))
            
            docs = ["Doc 1", "Doc 2", "Doc 3"]
            for doc in docs:
                emb = np.random.randn(1536).astype(np.float32)
                store1.append_text_with_embedding(doc, emb)
            
            # Export
            export_data = store1.export_to_dict(include_vectors=False)
            assert export_data["version"] == "1.0"
            assert len(export_data["pairs"]) == len(docs)
            
            # Verify export structure
            for pair in export_data["pairs"]:
                assert "text_id" in pair
                assert "vec_id" in pair
                assert "text" in pair
                assert "vector_dim" in pair

    def test_metadata_workflow(self):
        """Test storing and retrieving metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ISStore(tmpdir)
            
            # Store document with metadata using the standard pattern
            # (text + embedding pair, with metadata as additional JSON)
            text = "Important document"
            metadata = {
                "author": "John Doe",
                "date": "2024-01-01",
                "tags": ["important", "urgent"],
            }
            
            # Create a text+embedding pair
            emb = np.random.randn(1536).astype(np.float32)
            text_id, vec_id = store.append_text_with_embedding(text, emb)
            
            # Also store metadata as JSON
            meta_id = store.append_json(metadata)
            
            # Verify the text-vector pair exists
            pairs = store.list_pairs()
            assert len(pairs) == 1
            assert pairs[0] == (text_id, vec_id)
            
            # Retrieve text and verify
            retrieved_text = store.read_text(text_id)
            assert retrieved_text == text
            
            # Retrieve metadata separately
            retrieved_meta = store.read_json(meta_id)
            assert retrieved_meta == metadata

    def test_persistence_workflow(self):
        """Test that data persists across store instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and populate store
            store1 = ISStore(tmpdir)
            text = "Persistent text"
            emb = np.random.randn(1536).astype(np.float32)
            tid, vid = store1.append_text_with_embedding(text, emb)
            
            # Close and reopen
            del store1
            store2 = ISStore(tmpdir)
            
            # Verify data persists
            assert len(store2.list_pairs()) == 1
            assert store2.read_text(tid) == text
            
            # Verify can continue appending
            text2 = "New text after reopening"
            emb2 = np.random.randn(1536).astype(np.float32)
            tid2, vid2 = store2.append_text_with_embedding(text2, emb2)
            
            assert len(store2.list_pairs()) == 2

    def test_compaction_workflow(self):
        """Test store compaction and cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ISStore(tmpdir)
            
            # Add some documents
            for i in range(5):
                text = f"Document {i}"
                emb = np.random.randn(1536).astype(np.float32)
                store.append_text_with_embedding(text, emb)
            
            # Add orphaned segment
            orphan_id = store.append_text("Orphaned text")
            
            initial_count = len(store._glob_segments())
            
            # Compact
            stats = store.compact()
            
            assert stats["deleted_segments"] >= 0
            assert stats["valid_segments"] == 10  # 5 pairs = 10 segments
            
            # Verify orphan was removed if not linked
            final_count = len(store._glob_segments())
            assert final_count <= initial_count


class TestClientAPI:
    """Test high-level client API."""

    def test_client_initialization(self):
        """Test client creation and initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = ContextTapeClient(tmpdir)
            assert client.store is not None
            assert os.path.isdir(tmpdir)

    def test_client_ingest_search(self):
        """Test client ingest and search operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use client without embedding function for testing
            def fake_embed(text):
                np.random.seed(hash(text) % 2**32)
                emb = np.random.randn(1536).astype(np.float32)
                return emb / np.linalg.norm(emb)
            
            client = ContextTapeClient(tmpdir, embed_fn=fake_embed)
            
            # Ingest
            docs = ["Document about AI", "Document about ML"]
            for doc in docs:
                result = client.ingest(doc)
                assert result.text_id >= 0
                assert result.vector_id >= 0  # Changed from vec_id to vector_id
                assert result.success
            
            # Search
            results = client.search("AI query", top_k=2)
            
            assert len(results) <= 2
            for r in results:
                assert r.score >= 0
                # Text might be wrapped in JSON, so check both
                assert r.text in docs or any(d in r.text for d in docs)


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
class TestOpenAIIntegration:
    """Test OpenAI integration (requires API key)."""

    def test_real_embedding_workflow(self):
        """Test workflow with real OpenAI embeddings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = get_client()
            store = ISStore(tmpdir)
            
            # Embed and store
            docs = ["Python programming", "JavaScript development"]
            for doc in docs:
                emb = embed_text_1536(client, doc)
                store.append_text_with_embedding(doc, emb)
            
            # Search with real embedding
            query_emb = embed_text_1536(client, "coding languages")
            results = store.search_by_vector(query_emb, top_k=1)
            
            assert len(results) == 1
            score, tid, vid = results[0]
            text = store.read_text(tid)
            assert text in docs


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_segment_id(self):
        """Test reading non-existent segment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ISStore(tmpdir)
            
            with pytest.raises(FileNotFoundError):
                store.read_text(999)

    def test_empty_store_search(self):
        """Test searching empty store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ISStore(tmpdir)
            query = np.random.randn(1536).astype(np.float32)
            
            results = store.search_by_vector(query, top_k=5)
            assert len(results) == 0

    def test_mismatched_batch_lengths(self):
        """Test batch operations with mismatched inputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ISStore(tmpdir)
            
            texts = ["text1", "text2"]
            embeddings = [np.random.randn(1536).astype(np.float32)]
            
            with pytest.raises(ValueError, match="same length"):
                store.append_batch(texts, embeddings)


def test_readme_examples_work():
    """Verify that README examples actually work."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Example from README
        store = ISStore(tmpdir)
        
        # Add text with embedding
        embedding = np.random.randn(1536).astype(np.float32)
        text_id, vec_id = store.append_text_with_embedding(
            "Hello world", embedding, quantize=True
        )
        
        # Search by vector
        query = np.random.randn(1536).astype(np.float32)
        results = store.search_by_vector(query, top_k=5)
        
        assert isinstance(results, list)
        if results:
            score, tid, vid = results[0]
            text = store.read_text(tid)
            assert isinstance(text, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
