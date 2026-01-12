"""
ContextTape Integrations Module
===============================

This module provides ready-to-use integrations with popular ML/AI frameworks
and deployment patterns. Each integration is self-contained and can be used
as a starting point for your own projects.

Integrations:
    - FastAPI REST Server
    - LangChain Retriever
    - LlamaIndex Vector Store
    - Streaming/Async utilities
    - Flask Blueprint

Example:
    # FastAPI server
    from contexttape.integrations import create_fastapi_app
    app = create_fastapi_app("my_store")
    # Run: uvicorn module:app --reload

    # LangChain
    from contexttape.integrations import ContextTapeRetriever
    retriever = ContextTapeRetriever(store_path="my_store")
    docs = retriever.get_relevant_documents("my query")
"""

from __future__ import annotations

import os
import json
from typing import List, Dict, Any, Optional, Callable, Iterator
from dataclasses import dataclass, asdict
import numpy as np

from .storage import ISStore, MultiStore, DT_TEXT


# =============================================================================
# Data Classes for API responses
# =============================================================================

@dataclass
class SearchResult:
    """A single search result with metadata."""
    text: str
    score: float
    text_id: int
    vector_id: int
    store_path: str
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IngestResult:
    """Result of an ingestion operation."""
    text_id: int
    vector_id: int
    text_preview: str
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Core High-Level API
# =============================================================================

class ContextTapeClient:
    """
    High-level client for ContextTape operations.
    
    This provides a simplified interface for common operations like
    ingesting documents, searching, and managing stores.
    
    Example:
        >>> from contexttape.integrations import ContextTapeClient
        >>> 
        >>> client = ContextTapeClient("my_store")
        >>> 
        >>> # Ingest documents
        >>> client.ingest("The quick brown fox jumps over the lazy dog.")
        >>> client.ingest_batch(["Doc 1 content", "Doc 2 content", "Doc 3 content"])
        >>> 
        >>> # Search
        >>> results = client.search("quick fox", top_k=5)
        >>> for r in results:
        ...     print(f"{r.score:.4f}: {r.text[:50]}...")
        >>> 
        >>> # Stats
        >>> print(client.stats())
    """
    
    def __init__(
        self,
        store_path: str,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
        quantize: bool = False,
    ):
        """
        Initialize the ContextTape client.
        
        Args:
            store_path: Directory path for the segment store
            embed_fn: Custom embedding function (str -> np.ndarray).
                     If None, uses OpenAI text-embedding-3-small.
            quantize: Whether to use int8 quantization (4x smaller)
        """
        self.store = ISStore(store_path)
        self.store_path = store_path
        self.quantize = quantize
        
        if embed_fn is not None:
            self._embed = embed_fn
        else:
            # Lazy import to avoid requiring OpenAI key at import time
            self._embed = None
            self._client = None
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using configured embedding function."""
        if self._embed is not None:
            return self._embed(text)
        
        # Lazy initialize OpenAI client
        if self._client is None:
            from .embed import get_client, embed_text_1536
            self._client = get_client()
            self._embed_openai = lambda t: embed_text_1536(self._client, t)
        
        return self._embed_openai(text)
    
    def ingest(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IngestResult:
        """
        Ingest a single document into the store.
        
        Args:
            text: The text content to ingest
            metadata: Optional metadata dict (stored as JSON)
            
        Returns:
            IngestResult with IDs and status
        """
        try:
            # If metadata provided, wrap text in JSON structure
            if metadata:
                payload = json.dumps({
                    "text": text,
                    "meta": metadata
                }, ensure_ascii=False)
            else:
                payload = text
            
            embedding = self._get_embedding(text)
            tid, vid = self.store.append_text_with_embedding(
                payload, embedding, quantize=self.quantize
            )
            
            return IngestResult(
                text_id=tid,
                vector_id=vid,
                text_preview=text[:100],
                success=True
            )
        except Exception as e:
            return IngestResult(
                text_id=-1,
                vector_id=-1,
                text_preview=text[:100],
                success=False,
                error=str(e)
            )
    
    def ingest_batch(
        self,
        texts: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[IngestResult]:
        """
        Ingest multiple documents in batch.
        
        Args:
            texts: List of text contents
            metadata_list: Optional list of metadata dicts (one per text)
            progress_callback: Optional callback(current, total) for progress
            
        Returns:
            List of IngestResult objects
        """
        results = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            meta = metadata_list[i] if metadata_list and i < len(metadata_list) else None
            result = self.ingest(text, metadata=meta)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        return results
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        stride: int = 1,
        min_score: float = 0.0,
    ) -> List[SearchResult]:
        """
        Search the store for relevant documents.
        
        Args:
            query: The search query text
            top_k: Maximum number of results to return
            stride: Skip factor for faster (but less accurate) search
            min_score: Minimum cosine similarity threshold
            
        Returns:
            List of SearchResult objects sorted by score descending
        """
        query_vec = self._get_embedding(query)
        
        hits = self.store.search_by_vector(
            query_vec,
            top_k=top_k,
            stride=stride,
        )
        
        results = []
        for score, tid, eid in hits:
            if score < min_score:
                continue
                
            text = self.store.read_text(tid)
            
            # Try to parse metadata if JSON
            metadata = None
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict) and "text" in parsed:
                    text = parsed["text"]
                    metadata = parsed.get("meta")
            except (json.JSONDecodeError, TypeError):
                pass
            
            results.append(SearchResult(
                text=text,
                score=score,
                text_id=tid,
                vector_id=eid,
                store_path=self.store_path,
                metadata=metadata
            ))
        
        return results
    
    def stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        return self.store.stat()
    
    def __len__(self) -> int:
        """Return number of text-vector pairs in store."""
        return len(self.store.list_pairs())


# =============================================================================
# FastAPI Integration
# =============================================================================

def create_fastapi_app(
    store_path: str,
    title: str = "ContextTape API",
    embed_fn: Optional[Callable[[str], np.ndarray]] = None,
    quantize: bool = False,
):
    """
    Create a FastAPI application for ContextTape.
    
    Example:
        >>> from contexttape.integrations import create_fastapi_app
        >>> app = create_fastapi_app("my_store")
        >>> # Run with: uvicorn module:app --host 0.0.0.0 --port 8000
    
    Endpoints:
        POST /ingest - Ingest a document
        POST /ingest/batch - Ingest multiple documents
        POST /search - Search for documents
        GET /stats - Get store statistics
        GET /health - Health check
    
    Args:
        store_path: Path to segment store directory
        title: API title for docs
        embed_fn: Optional custom embedding function
        quantize: Whether to use int8 quantization
        
    Returns:
        FastAPI application instance
    """
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        from typing import List, Optional
    except ImportError:
        raise ImportError(
            "FastAPI integration requires: pip install fastapi uvicorn pydantic"
        )
    
    app = FastAPI(title=title, description="ContextTape RAG Storage API")
    client = ContextTapeClient(store_path, embed_fn=embed_fn, quantize=quantize)
    
    # Request/Response models
    class IngestRequest(BaseModel):
        text: str
        metadata: Optional[Dict[str, Any]] = None
    
    class BatchIngestRequest(BaseModel):
        texts: List[str]
        metadata_list: Optional[List[Dict[str, Any]]] = None
    
    class SearchRequest(BaseModel):
        query: str
        top_k: int = 5
        min_score: float = 0.0
        stride: int = 1
    
    @app.post("/ingest")
    async def ingest_endpoint(req: IngestRequest):
        """Ingest a single document."""
        result = client.ingest(req.text, metadata=req.metadata)
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)
        return result.to_dict()
    
    @app.post("/ingest/batch")
    async def batch_ingest_endpoint(req: BatchIngestRequest):
        """Ingest multiple documents."""
        results = client.ingest_batch(req.texts, metadata_list=req.metadata_list)
        return {
            "results": [r.to_dict() for r in results],
            "success_count": sum(1 for r in results if r.success),
            "failure_count": sum(1 for r in results if not r.success),
        }
    
    @app.post("/search")
    async def search_endpoint(req: SearchRequest):
        """Search for relevant documents."""
        results = client.search(
            req.query,
            top_k=req.top_k,
            min_score=req.min_score,
            stride=req.stride,
        )
        return {
            "results": [r.to_dict() for r in results],
            "count": len(results),
        }
    
    @app.get("/stats")
    async def stats_endpoint():
        """Get store statistics."""
        return client.stats()
    
    @app.get("/health")
    async def health_endpoint():
        """Health check."""
        return {"status": "ok", "store": store_path, "pairs": len(client)}
    
    return app


# =============================================================================
# LangChain Integration
# =============================================================================

class ContextTapeRetriever:
    """
    LangChain-compatible retriever for ContextTape.
    
    Example:
        >>> from contexttape.integrations import ContextTapeRetriever
        >>> from langchain.chains import RetrievalQA
        >>> from langchain.llms import OpenAI
        >>> 
        >>> retriever = ContextTapeRetriever("my_store", k=5)
        >>> 
        >>> # Use directly
        >>> docs = retriever.get_relevant_documents("What is photosynthesis?")
        >>> 
        >>> # Or with LangChain chain
        >>> qa = RetrievalQA.from_chain_type(
        ...     llm=OpenAI(),
        ...     retriever=retriever,
        ... )
        >>> answer = qa.run("What is photosynthesis?")
    """
    
    def __init__(
        self,
        store_path: str,
        k: int = 5,
        min_score: float = 0.0,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
    ):
        """
        Initialize the retriever.
        
        Args:
            store_path: Path to segment store
            k: Number of documents to retrieve
            min_score: Minimum similarity score threshold
            embed_fn: Optional custom embedding function
        """
        self.client = ContextTapeClient(store_path, embed_fn=embed_fn)
        self.k = k
        self.min_score = min_score
    
    def get_relevant_documents(self, query: str) -> List[Any]:
        """
        Retrieve relevant documents for a query.
        
        Returns LangChain Document objects if langchain is installed,
        otherwise returns dicts with page_content and metadata.
        """
        results = self.client.search(query, top_k=self.k, min_score=self.min_score)
        
        try:
            from langchain.schema import Document
            return [
                Document(
                    page_content=r.text,
                    metadata={
                        "score": r.score,
                        "text_id": r.text_id,
                        "vector_id": r.vector_id,
                        **(r.metadata or {})
                    }
                )
                for r in results
            ]
        except ImportError:
            # Return dict-like objects if langchain not installed
            return [
                {
                    "page_content": r.text,
                    "metadata": {
                        "score": r.score,
                        "text_id": r.text_id,
                        "vector_id": r.vector_id,
                        **(r.metadata or {})
                    }
                }
                for r in results
            ]
    
    async def aget_relevant_documents(self, query: str) -> List[Any]:
        """Async version of get_relevant_documents."""
        return self.get_relevant_documents(query)


# =============================================================================
# LlamaIndex Integration  
# =============================================================================

class ContextTapeVectorStore:
    """
    LlamaIndex-compatible vector store for ContextTape.
    
    Example:
        >>> from contexttape.integrations import ContextTapeVectorStore
        >>> from llama_index import VectorStoreIndex, SimpleDirectoryReader
        >>> 
        >>> # Create vector store
        >>> vector_store = ContextTapeVectorStore("my_store")
        >>> 
        >>> # Build index from documents
        >>> documents = SimpleDirectoryReader("data").load_data()
        >>> index = VectorStoreIndex.from_documents(
        ...     documents,
        ...     vector_store=vector_store,
        ... )
        >>> 
        >>> # Query
        >>> query_engine = index.as_query_engine()
        >>> response = query_engine.query("What is the main topic?")
    """
    
    def __init__(
        self,
        store_path: str,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
        quantize: bool = False,
    ):
        self.client = ContextTapeClient(store_path, embed_fn=embed_fn, quantize=quantize)
        self.store_path = store_path
    
    def add(self, nodes: List[Any]) -> List[str]:
        """Add nodes to the vector store."""
        ids = []
        for node in nodes:
            # Handle LlamaIndex nodes
            text = getattr(node, 'text', None) or getattr(node, 'get_content', lambda: str(node))()
            metadata = getattr(node, 'metadata', {})
            
            result = self.client.ingest(text, metadata=metadata)
            ids.append(f"{result.text_id}:{result.vector_id}")
        
        return ids
    
    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete is not directly supported (append-only store)."""
        raise NotImplementedError(
            "ContextTape uses append-only storage. "
            "To remove documents, create a new store without them."
        )
    
    def query(
        self,
        query_embedding: List[float],
        similarity_top_k: int = 5,
        **kwargs: Any,
    ) -> List[Any]:
        """Query the vector store with an embedding."""
        query_vec = np.array(query_embedding, dtype=np.float32)
        
        hits = self.client.store.search_by_vector(query_vec, top_k=similarity_top_k)
        
        results = []
        for score, tid, eid in hits:
            text = self.client.store.read_text(tid)
            results.append({
                "text": text,
                "score": score,
                "id": f"{tid}:{eid}",
            })
        
        return results


# =============================================================================
# Streaming / Generator Utilities
# =============================================================================

def stream_search(
    store: ISStore,
    query_vec: np.ndarray,
    min_score: float = 0.0,
    batch_size: int = 100,
) -> Iterator[Tuple[float, int, int, str]]:
    """
    Stream search results as they are found.
    
    Useful for very large stores where you want results as they're discovered.
    
    Args:
        store: The ISStore to search
        query_vec: Query embedding vector
        min_score: Minimum cosine similarity to yield
        batch_size: Number of vectors to scan per batch
        
    Yields:
        Tuples of (score, text_id, vector_id, text_content)
        
    Example:
        >>> for score, tid, vid, text in stream_search(store, query_vec, min_score=0.5):
        ...     print(f"{score:.4f}: {text[:50]}...")
        ...     if some_condition:
        ...         break  # Early termination
    """
    pairs = store.list_pairs()
    
    from .storage import ISStore
    
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        
        for tid, eid in batch:
            try:
                vec = store.read_vector(eid)
                score = ISStore._cosine(query_vec, vec)
                
                if score >= min_score:
                    text = store.read_text(tid)
                    yield (score, tid, eid, text)
            except Exception:
                continue


def iterate_store(store: ISStore) -> Iterator[Tuple[int, int, str, np.ndarray]]:
    """
    Iterate over all text-vector pairs in a store.
    
    Args:
        store: The ISStore to iterate
        
    Yields:
        Tuples of (text_id, vector_id, text_content, embedding_vector)
        
    Example:
        >>> for tid, vid, text, vec in iterate_store(store):
        ...     print(f"[{tid}] {text[:50]}... (dim={vec.shape[0]})")
    """
    for tid, vid in store.list_pairs():
        try:
            text = store.read_text(tid)
            vec = store.read_vector(vid)
            yield (tid, vid, text, vec)
        except Exception:
            continue


# =============================================================================
# Export utilities
# =============================================================================

def export_to_jsonl(store: ISStore, output_path: str) -> int:
    """
    Export store contents to JSONL format.
    
    Args:
        store: The store to export
        output_path: Path to write JSONL file
        
    Returns:
        Number of records exported
        
    Example:
        >>> count = export_to_jsonl(store, "backup.jsonl")
        >>> print(f"Exported {count} records")
    """
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for tid, vid, text, vec in iterate_store(store):
            record = {
                "text_id": tid,
                "vector_id": vid,
                "text": text,
                "embedding": vec.tolist(),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    
    return count


def import_from_jsonl(
    store: ISStore,
    input_path: str,
    quantize: bool = False,
) -> int:
    """
    Import records from JSONL format.
    
    Args:
        store: The store to import into
        input_path: Path to JSONL file
        quantize: Whether to use int8 quantization
        
    Returns:
        Number of records imported
    """
    count = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            
            record = json.loads(line)
            text = record["text"]
            vec = np.array(record["embedding"], dtype=np.float32)
            
            store.append_text_with_embedding(text, vec, quantize=quantize)
            count += 1
    
    return count


# =============================================================================
# Convenience function to check if common deps are available
# =============================================================================

def check_integrations() -> Dict[str, bool]:
    """
    Check which integration dependencies are available.
    
    Returns:
        Dict mapping integration name to availability boolean
    """
    checks = {}
    
    try:
        import fastapi
        checks["fastapi"] = True
    except ImportError:
        checks["fastapi"] = False
    
    try:
        import langchain
        checks["langchain"] = True
    except ImportError:
        checks["langchain"] = False
    
    try:
        import llama_index
        checks["llama_index"] = True
    except ImportError:
        checks["llama_index"] = False
    
    try:
        import openai
        checks["openai"] = True
    except ImportError:
        checks["openai"] = False
    
    return checks
