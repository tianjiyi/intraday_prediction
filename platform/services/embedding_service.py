"""
Embedding Service for Agent Memory Vector Store.

Converts text to 384-dimensional vectors using sentence-transformers
for semantic similarity search in pgvector.
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Lazy import to avoid startup penalty if not used
SentenceTransformer = None


class EmbeddingService:
    """Converts text to vector embeddings for semantic memory search."""

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        memory_config = config.get("memory", {})
        self.model_name = memory_config.get("embedding_model", self.DEFAULT_MODEL)
        self.model = None
        self._init_model()

    def _init_model(self):
        """Load sentence-transformers model (graceful if unavailable)."""
        try:
            from sentence_transformers import SentenceTransformer as ST
            self.model = ST(self.model_name)
            logger.info(f"Embedding model loaded: {self.model_name} ({self.EMBEDDING_DIM} dims)")
        except ImportError:
            logger.warning("sentence-transformers not installed. Embedding service unavailable.")
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")

    def is_available(self) -> bool:
        """Check if the embedding model is loaded."""
        return self.model is not None

    def embed(self, text: str) -> List[float]:
        """Convert a single text to an embedding vector (384 dims)."""
        if not self.is_available():
            raise RuntimeError("Embedding model not available")
        return self.model.encode(text).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Convert multiple texts to embeddings in a single batch."""
        if not self.is_available():
            raise RuntimeError("Embedding model not available")
        return [v.tolist() for v in self.model.encode(texts)]
