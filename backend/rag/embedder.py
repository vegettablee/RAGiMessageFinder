"""
Shared embedder module to ensure SentenceTransformer is only loaded once.
"""
from sentence_transformers import SentenceTransformer, SimilarityFunction

# Initialize embedders once and share across all modules
_chunking_embedder = None
_rag_embedder = None


def get_chunking_embedder():
    """Get the embedder for chunking (all-MiniLM-L6-v2)."""
    global _chunking_embedder
    if _chunking_embedder is None:
        print("Loading chunking embedder (all-MiniLM-L6-v2)...")
        _chunking_embedder = SentenceTransformer(
            "all-MiniLM-L6-v2",
            similarity_fn_name=SimilarityFunction.COSINE
        )
    return _chunking_embedder


def get_rag_embedder():
    """Get the embedder for RAG queries (multi-qa-MiniLM-L6-cos-v1)."""
    global _rag_embedder
    if _rag_embedder is None:
        print("Loading RAG embedder (multi-qa-MiniLM-L6-cos-v1)...")
        _rag_embedder = SentenceTransformer(
            "multi-qa-MiniLM-L6-cos-v1",
            similarity_fn_name=SimilarityFunction.COSINE
        )
    return _rag_embedder
