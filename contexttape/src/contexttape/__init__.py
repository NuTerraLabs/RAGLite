# src/contexttape/__init__.py
"""
ContextTape: Zero-infrastructure RAG storage.

A new datatype for storing text + embeddings in tiny .is segment files.
No vector database required. Just files.

Quick Example:
    >>> from contexttape import ISStore
    >>> store = ISStore("my_store")
    >>> store.append_text_with_embedding("Hello world", embedding_vector)
    >>> hits = store.search_by_vector(query_vector, top_k=5)

For embedding generation, use with OpenAI:
    >>> from contexttape import get_client, embed_text_1536
    >>> client = get_client()  # requires OPENAI_API_KEY
    >>> vec = embed_text_1536(client, "Your text here")

High-Level Client API:
    >>> from contexttape import ContextTapeClient
    >>> client = ContextTapeClient("my_store")
    >>> client.ingest("Document text here")
    >>> results = client.search("query", top_k=5)

Integrations:
    - FastAPI: create_fastapi_app("store_path")
    - LangChain: ContextTapeRetriever("store_path")
    - LlamaIndex: ContextTapeVectorStore("store_path")
"""

__version__ = "0.5.0"

# Core storage
from .storage import (
    ISStore,
    MultiStore,
    ISHeader,
    # Data type constants
    DT_TEXT,
    DT_VEC_F32,
    DT_VEC_I8,
    DT_COARSE,
    DT_IMAGE,
    DT_AUDIO,
    DT_BLOB,
    DT_JSON,
    # Header size
    HEADER_SIZE,
    # Utility
    write_playlist,
)

# Embedding utilities
from .embed import (
    get_client,
    embed_text_1536,
)

# Search utilities
from .search import (
    combined_search,
    lexical_overlap,
    hybrid_score,
)

# Relevance/selection
from .relevance import select_relevant_blocks

# High-level integrations
from .integrations import (
    ContextTapeClient,
    SearchResult,
    IngestResult,
    create_fastapi_app,
    ContextTapeRetriever,
    ContextTapeVectorStore,
    stream_search,
    iterate_store,
    export_to_jsonl,
    import_from_jsonl,
    check_integrations,
)

__all__ = [
    # Version
    "__version__",
    # Core classes
    "ISStore",
    "MultiStore", 
    "ISHeader",
    # Data types
    "DT_TEXT",
    "DT_VEC_F32",
    "DT_VEC_I8",
    "DT_COARSE",
    "DT_IMAGE",
    "DT_AUDIO",
    "DT_BLOB",
    "DT_JSON",
    "HEADER_SIZE",
    # Storage functions
    "write_playlist",
    # Embedding
    "get_client",
    "embed_text_1536",
    # Search
    "combined_search",
    "lexical_overlap",
    "hybrid_score",
    "select_relevant_blocks",
    # High-level API
    "ContextTapeClient",
    "SearchResult",
    "IngestResult",
    # Framework integrations
    "create_fastapi_app",
    "ContextTapeRetriever",
    "ContextTapeVectorStore",
    # Utilities
    "stream_search",
    "iterate_store",
    "export_to_jsonl",
    "import_from_jsonl",
    "check_integrations",
]
