"""
Comprehensive Tests for ContextTape Storage System
==================================================

This test suite covers:
- ISHeader serialization/deserialization
- ISStore CRUD operations
- Vector storage (f32 and i8 quantized)
- Search functionality
- MultiStore late fusion
- Integration components
- Edge cases and error handling
"""
import tempfile
import shutil
import json
import os
import numpy as np
import pytest

from contexttape import (
    ISStore,
    MultiStore,
    ISHeader,
    DT_TEXT,
    DT_VEC_F32,
    DT_VEC_I8,
    DT_JSON,
    DT_BLOB,
    HEADER_SIZE,
    lexical_overlap,
    hybrid_score,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_store():
    """Create a temporary store directory."""
    tmpdir = tempfile.mkdtemp(prefix="contexttape_test_")
    store = ISStore(tmpdir)
    yield store
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def populated_store():
    """Create a store with sample data."""
    tmpdir = tempfile.mkdtemp(prefix="contexttape_populated_")
    store = ISStore(tmpdir)
    
    docs = [
        "Machine learning enables computers to learn from data.",
        "Neural networks are inspired by biological neurons.",
        "Deep learning uses multiple layers for feature extraction.",
        "Natural language processing handles human language.",
        "Computer vision interprets visual information.",
    ]
    
    for doc in docs:
        # Create deterministic embeddings for testing
        np.random.seed(hash(doc) % 2**32)
        vec = np.random.randn(1536).astype(np.float32)
        vec = vec / np.linalg.norm(vec)  # Normalize
        store.append_text_with_embedding(doc, vec)
    
    yield store
    shutil.rmtree(tmpdir, ignore_errors=True)


# =============================================================================
# ISHeader Tests
# =============================================================================

class TestISHeader:
    """Tests for ISHeader class."""

    def test_header_pack_unpack_roundtrip(self):
        """Test that header packing and unpacking is lossless."""
        original = ISHeader(
            next_id=42,
            prev_id=10,
            data_len=1024,
            data_type=DT_VEC_F32,
            dim=1536,
            scale=0.5,
            ts=1704067200,
        )
        packed = original.pack()
        assert len(packed) == HEADER_SIZE
        
        unpacked = ISHeader.unpack(packed)
        assert unpacked.next_id == 42
        assert unpacked.prev_id == 10
        assert unpacked.data_len == 1024
        assert unpacked.data_type == DT_VEC_F32
        assert unpacked.dim == 1536
        assert abs(unpacked.scale - 0.5) < 1e-6
        assert unpacked.ts == 1704067200

    def test_header_defaults(self):
        """Test header with default values."""
        h = ISHeader()
        assert h.next_id == -1
        assert h.prev_id == -1
        assert h.data_len == 0
        assert h.data_type == DT_TEXT
        assert h.dim == 0
        assert h.scale == 1.0

    def test_header_negative_ids(self):
        """Test headers handle negative IDs correctly."""
        h = ISHeader(next_id=-1, prev_id=-1)
        packed = h.pack()
        unpacked = ISHeader.unpack(packed)
        assert unpacked.next_id == -1
        assert unpacked.prev_id == -1

    def test_header_large_values(self):
        """Test headers handle large values."""
        h = ISHeader(
            next_id=2**30,
            prev_id=2**30,
            data_len=2**30,
            dim=2**20,
        )
        packed = h.pack()
        unpacked = ISHeader.unpack(packed)
        assert unpacked.next_id == 2**30
        assert unpacked.data_len == 2**30

    def test_header_timestamp_auto(self):
        """Test that timestamp is auto-generated if not provided."""
        import time
        before = int(time.time())
        h = ISHeader()
        after = int(time.time())
        assert before <= h.ts <= after


# =============================================================================
# ISStore Basic Operations Tests
# =============================================================================

class TestISStoreBasics:
    """Tests for basic ISStore operations."""

    def test_store_creation(self, temp_store):
        """Test store directory creation."""
        assert os.path.isdir(temp_store.dir_path)
        assert temp_store.next_id == 0

    def test_append_text(self, temp_store):
        """Test appending text segments."""
        text = "Hello, ContextTape!"
        seg_id = temp_store.append_text(text)
        
        assert seg_id == 0
        retrieved = temp_store.read_text(seg_id)
        assert retrieved == text

    def test_append_multiple_texts(self, temp_store):
        """Test appending multiple text segments."""
        texts = ["First", "Second", "Third"]
        ids = [temp_store.append_text(t) for t in texts]
        
        assert ids == [0, 1, 2]
        for i, text in enumerate(texts):
            assert temp_store.read_text(i) == text

    def test_append_unicode_text(self, temp_store):
        """Test Unicode text handling."""
        texts = [
            "Hello 世界",
            "Привет мир",
            "مرحبا بالعالم",
            "🎉🚀🔥",
        ]
        for text in texts:
            tid = temp_store.append_text(text)
            assert temp_store.read_text(tid) == text

    def test_append_large_text(self, temp_store):
        """Test large text handling."""
        text = "x" * 1_000_000  # 1MB of text
        tid = temp_store.append_text(text)
        retrieved = temp_store.read_text(tid)
        assert len(retrieved) == 1_000_000
        assert retrieved == text

    def test_store_persistence(self):
        """Test that store persists across instances."""
        tmpdir = tempfile.mkdtemp(prefix="contexttape_persist_")
        try:
            # Create and populate
            store1 = ISStore(tmpdir)
            store1.append_text("Persistent text")
            
            # Reopen
            store2 = ISStore(tmpdir)
            assert store2.read_text(0) == "Persistent text"
            assert store2.next_id == 1
        finally:
            shutil.rmtree(tmpdir)


# =============================================================================
# Vector Storage Tests
# =============================================================================

class TestVectorStorage:
    """Tests for vector storage functionality."""

    def test_append_vector_f32(self, temp_store):
        """Test appending float32 vectors."""
        vec = np.random.randn(1536).astype(np.float32)
        seg_id = temp_store.append_vector_f32(vec)
        
        retrieved = temp_store.read_vector(seg_id)
        np.testing.assert_array_almost_equal(vec, retrieved, decimal=5)

    def test_append_vector_different_dims(self, temp_store):
        """Test vectors of different dimensions."""
        dims = [128, 384, 768, 1536, 3072]
        for dim in dims:
            vec = np.random.randn(dim).astype(np.float32)
            seg_id = temp_store.append_vector_f32(vec)
            retrieved = temp_store.read_vector(seg_id)
            assert retrieved.shape[0] == dim
            np.testing.assert_array_almost_equal(vec, retrieved, decimal=5)

    def test_append_vector_i8_quantization(self, temp_store):
        """Test int8 quantization preserves approximate values."""
        vec = np.random.randn(1536).astype(np.float32)
        seg_id = temp_store.append_vector_i8(vec)
        
        retrieved = temp_store.read_vector(seg_id)
        # Should be close but not exact due to quantization
        assert retrieved.shape == vec.shape
        # Correlation should be very high
        correlation = np.corrcoef(vec, retrieved)[0, 1]
        assert correlation > 0.99

    def test_quantization_preserves_direction(self, temp_store):
        """Test that quantization preserves vector direction (important for cosine)."""
        vec = np.random.randn(1536).astype(np.float32)
        vec_norm = vec / np.linalg.norm(vec)
        
        seg_id = temp_store.append_vector_i8(vec)
        retrieved = temp_store.read_vector(seg_id)
        retrieved_norm = retrieved / np.linalg.norm(retrieved)
        
        # Cosine similarity should be very high
        cosine_sim = np.dot(vec_norm, retrieved_norm)
        assert cosine_sim > 0.995

    def test_append_text_with_embedding(self, temp_store):
        """Test adding paired text + embedding."""
        text = "Neural networks are function approximators."
        vec = np.random.randn(1536).astype(np.float32)
        
        t_id, v_id = temp_store.append_text_with_embedding(text, vec)
        
        assert t_id == 0
        assert v_id == 1
        assert temp_store.read_text(t_id) == text
        np.testing.assert_array_almost_equal(
            vec, temp_store.read_vector(v_id), decimal=5
        )

    def test_quantized_text_with_embedding(self, temp_store):
        """Test quantized vector storage."""
        text = "Quantized document"
        vec = np.random.randn(1536).astype(np.float32)
        
        t_id, v_id = temp_store.append_text_with_embedding(
            text, vec, quantize=True
        )
        
        # Check the header indicates int8
        header = temp_store._read_header(v_id)
        assert header.data_type == DT_VEC_I8


# =============================================================================
# Linking and Pairs Tests
# =============================================================================

class TestLinkingAndPairs:
    """Tests for text-vector linking."""

    def test_link_text_to_vec(self, temp_store):
        """Test explicit linking of text to vector."""
        text_id = temp_store.append_text("Test text")
        vec_id = temp_store.append_vector_f32(
            np.random.randn(1536).astype(np.float32)
        )
        
        temp_store.link_text_to_vec(text_id, vec_id)
        
        header = temp_store._read_header(text_id)
        assert header.next_id == vec_id

    def test_list_pairs(self, temp_store):
        """Test listing text-vector pairs."""
        texts = ["Doc one", "Doc two", "Doc three"]
        
        for text in texts:
            vec = np.random.randn(1536).astype(np.float32)
            temp_store.append_text_with_embedding(text, vec)
        
        pairs = temp_store.list_pairs()
        assert len(pairs) == 3
        assert pairs == [(0, 1), (2, 3), (4, 5)]

    def test_list_segments_by_type(self, temp_store):
        """Test listing segments filtered by type."""
        # Add mixed content
        temp_store.append_text("Text 1")
        temp_store.append_vector_f32(np.random.randn(100).astype(np.float32))
        temp_store.append_text("Text 2")
        temp_store.append_vector_i8(np.random.randn(100).astype(np.float32))
        
        text_segs = temp_store.list_segments(DT_TEXT)
        vec_f32_segs = temp_store.list_segments(DT_VEC_F32)
        vec_i8_segs = temp_store.list_segments(DT_VEC_I8)
        all_segs = temp_store.list_segments()
        
        assert len(text_segs) == 2
        assert len(vec_f32_segs) == 1
        assert len(vec_i8_segs) == 1
        assert len(all_segs) == 4


# =============================================================================
# Search Tests
# =============================================================================

class TestSearch:
    """Tests for search functionality."""

    def test_search_by_vector(self, populated_store):
        """Test vector search returns ranked results."""
        # Use the same seed as first document
        np.random.seed(hash("Machine learning enables computers to learn from data.") % 2**32)
        query_vec = np.random.randn(1536).astype(np.float32)
        query_vec = query_vec / np.linalg.norm(query_vec)
        
        results = populated_store.search_by_vector(query_vec, top_k=3)
        
        assert len(results) == 3
        # First result should be exact match (score ~1.0)
        score, tid, _ = results[0]
        assert score > 0.99
        assert "Machine learning" in populated_store.read_text(tid)

    def test_search_top_k_limit(self, populated_store):
        """Test that top_k limits results."""
        query_vec = np.random.randn(1536).astype(np.float32)
        
        for k in [1, 2, 3, 5, 10]:
            results = populated_store.search_by_vector(query_vec, top_k=k)
            assert len(results) <= k

    def test_search_stride(self, populated_store):
        """Test stride parameter skips segments."""
        query_vec = np.random.randn(1536).astype(np.float32)
        
        results_no_stride = populated_store.search_by_vector(query_vec, top_k=10, stride=1)
        results_stride_2 = populated_store.search_by_vector(query_vec, top_k=10, stride=2)
        
        # With stride=2, we scan half the vectors
        assert len(results_stride_2) <= len(results_no_stride)

    def test_search_empty_store(self, temp_store):
        """Test search on empty store."""
        query_vec = np.random.randn(1536).astype(np.float32)
        results = temp_store.search_by_vector(query_vec, top_k=5)
        assert results == []

    def test_cosine_similarity_static(self):
        """Test cosine similarity computation."""
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        c = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        d = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        
        assert abs(ISStore._cosine(a, b) - 1.0) < 1e-6  # Same direction
        assert abs(ISStore._cosine(a, c) - 0.0) < 1e-6  # Orthogonal
        assert abs(ISStore._cosine(a, d) + 1.0) < 1e-6  # Opposite


# =============================================================================
# MultiStore Tests
# =============================================================================

class TestMultiStore:
    """Tests for MultiStore (multi-directory search)."""

    def test_multi_store_search(self):
        """Test searching across multiple stores."""
        tmpdir1 = tempfile.mkdtemp(prefix="contexttape_test1_")
        tmpdir2 = tempfile.mkdtemp(prefix="contexttape_test2_")
        
        try:
            store1 = ISStore(tmpdir1)
            store2 = ISStore(tmpdir2)
            
            # Add different docs to each store
            vec1 = np.array([1.0] * 1536, dtype=np.float32)
            vec1 = vec1 / np.linalg.norm(vec1)
            store1.append_text_with_embedding("Store 1 doc", vec1)
            
            vec2 = np.array([2.0] * 1536, dtype=np.float32)
            vec2 = vec2 / np.linalg.norm(vec2)
            store2.append_text_with_embedding("Store 2 doc", vec2)
            
            # Search across both
            ms = MultiStore([store1, store2])
            query = np.array([1.0] * 1536, dtype=np.float32)
            query = query / np.linalg.norm(query)
            
            results = ms.search(query, per_shard_k=2, final_k=2)
            assert len(results) == 2
            
            # First result should be from store1 (exact match)
            assert results[0][0] == tmpdir1
            
        finally:
            shutil.rmtree(tmpdir1, ignore_errors=True)
            shutil.rmtree(tmpdir2, ignore_errors=True)

    def test_multi_store_empty_stores(self):
        """Test MultiStore with some empty stores."""
        tmpdir1 = tempfile.mkdtemp(prefix="contexttape_empty1_")
        tmpdir2 = tempfile.mkdtemp(prefix="contexttape_empty2_")
        
        try:
            store1 = ISStore(tmpdir1)  # Empty
            store2 = ISStore(tmpdir2)
            
            vec = np.random.randn(1536).astype(np.float32)
            store2.append_text_with_embedding("Only doc", vec)
            
            ms = MultiStore([store1, store2])
            results = ms.search(vec, per_shard_k=5, final_k=5)
            
            assert len(results) == 1
            assert results[0][0] == tmpdir2
            
        finally:
            shutil.rmtree(tmpdir1, ignore_errors=True)
            shutil.rmtree(tmpdir2, ignore_errors=True)


# =============================================================================
# JSON and Bytes Storage Tests
# =============================================================================

class TestJSONAndBytes:
    """Tests for JSON and bytes segment storage."""

    def test_json_segment(self, temp_store):
        """Test JSON segment storage."""
        data = {"name": "test", "values": [1, 2, 3], "nested": {"a": 1}}
        seg_id = temp_store.append_json(data)
        
        retrieved = temp_store.read_json(seg_id)
        assert retrieved == data

    def test_json_unicode(self, temp_store):
        """Test JSON with Unicode."""
        data = {"message": "Hello 世界 🌍", "chars": ["α", "β", "γ"]}
        seg_id = temp_store.append_json(data)
        
        retrieved = temp_store.read_json(seg_id)
        assert retrieved == data

    def test_bytes_segment(self, temp_store):
        """Test raw bytes storage."""
        data = b"\x00\x01\x02\x03\xff\xfe\xfd"
        seg_id = temp_store.append_bytes(data)
        
        retrieved = temp_store.read_bytes(seg_id)
        assert retrieved == data

    def test_large_bytes(self, temp_store):
        """Test large binary storage."""
        data = os.urandom(100_000)  # 100KB random bytes
        seg_id = temp_store.append_bytes(data)
        
        retrieved = temp_store.read_bytes(seg_id)
        assert retrieved == data

    def test_media_with_manifest(self, temp_store):
        """Test media blob with manifest."""
        media_bytes = b"fake image data here"
        
        man_id, blob_id = temp_store.append_media_with_manifest(
            media_bytes,
            media_kind="image",
            filename="test.png",
            meta={"width": 100, "height": 100}
        )
        
        # Read manifest
        manifest = temp_store.read_json(man_id)
        assert manifest["type"] == "image"
        assert manifest["filename"] == "test.png"
        assert manifest["ref_blob_id"] == blob_id
        
        # Read blob
        blob = temp_store.read_bytes(blob_id)
        assert blob == media_bytes


# =============================================================================
# Hybrid Search Tests
# =============================================================================

class TestHybridSearch:
    """Tests for hybrid search functionality."""

    def test_lexical_overlap(self):
        """Test lexical overlap scoring."""
        query = "quantum computing applications"
        
        # High overlap
        text1 = "Quantum computing has many applications in science."
        assert lexical_overlap(query, text1) > 0.5
        
        # No overlap
        text2 = "The weather is nice today."
        assert lexical_overlap(query, text2) == 0.0

    def test_lexical_overlap_case_insensitive(self):
        """Test that lexical overlap is case-insensitive."""
        query = "Machine Learning"
        text = "machine learning is great"
        
        overlap = lexical_overlap(query, text)
        assert overlap > 0.5

    def test_hybrid_score(self):
        """Test hybrid scoring formula."""
        # Pure vector (alpha=1)
        assert hybrid_score(1.0, cos=0.8, lex=0.2) == 0.8
        
        # Pure lexical (alpha=0)
        assert hybrid_score(0.0, cos=0.8, lex=0.2) == 0.2
        
        # Mixed (alpha=0.5)
        assert abs(hybrid_score(0.5, cos=0.8, lex=0.2) - 0.5) < 1e-6

    def test_hybrid_score_bounds(self):
        """Test hybrid score stays in [0, 1] range."""
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for cos in [0.0, 0.5, 1.0]:
                for lex in [0.0, 0.5, 1.0]:
                    score = hybrid_score(alpha, cos, lex)
                    assert 0.0 <= score <= 1.0


# =============================================================================
# Stats Tests
# =============================================================================

class TestStats:
    """Tests for store statistics."""

    def test_stat_empty(self, temp_store):
        """Test stats on empty store."""
        stats = temp_store.stat()
        assert stats["text_segments"] == 0
        assert stats["vector_segments"] == 0
        assert stats["pairs"] == 0
        assert stats["next_id"] == 0

    def test_stat_populated(self, populated_store):
        """Test stats on populated store."""
        stats = populated_store.stat()
        assert stats["text_segments"] == 5
        assert stats["vector_segments"] == 5
        assert stats["pairs"] == 5
        assert stats["next_id"] == 10


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    def test_read_nonexistent_segment(self, temp_store):
        """Test reading non-existent segment raises error."""
        with pytest.raises(FileNotFoundError):
            temp_store.read_text(999)

    def test_read_wrong_type(self, temp_store):
        """Test reading segment as wrong type raises error."""
        # Add text segment
        tid = temp_store.append_text("Hello")
        
        # Try to read as vector
        with pytest.raises(ValueError, match="Not a vector segment"):
            temp_store.read_vector(tid)

    def test_read_text_from_vector(self, temp_store):
        """Test reading vector segment as text raises error."""
        vid = temp_store.append_vector_f32(np.random.randn(100).astype(np.float32))
        
        with pytest.raises(ValueError, match="Not a text segment"):
            temp_store.read_text(vid)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
