"""
Embedding service using Sentence Transformers with Qwen3-embedding-0.6B model
"""
from __future__ import annotations

import os
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Wrapper for Sentence Transformers embedding model"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        cache_folder: Optional[str] = None,
        local_model_path: Optional[str] = None,
    ):
        """
        Initialize the embedding model.

        Args:
            model_name: HuggingFace model identifier (used only if local_model_path is absent)
            device: Device to run on ('cuda', 'cpu', 'mps'). Auto-detected if None.
            normalize_embeddings: Whether to normalize embeddings to unit length
            cache_folder: Custom cache folder for model files (HuggingFace download only)
            local_model_path: Absolute path to a locally stored model directory.
                              When provided and the path exists, the model is loaded from
                              disk without any network access.
        """
        self.normalize_embeddings = normalize_embeddings
        self._local_model_path = local_model_path

        # Determine effective model source: local path takes priority
        if local_model_path and os.path.isdir(local_model_path):
            self.model_name = local_model_path
            logger.info(f"Using local model path: {local_model_path}")
        else:
            if local_model_path:
                logger.warning(
                    f"LOCAL_MODEL_PATH '{local_model_path}' does not exist or is not a directory, "
                    "falling back to HuggingFace download."
                )
            self.model_name = model_name

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self._model: Optional[SentenceTransformer] = None
        self._cache_folder = cache_folder

        logger.info(f"EmbeddingModel initialized with model={self.model_name}, device={device}")

    @property
    def model(self) -> SentenceTransformer:
        """Lazy loading of the model"""
        if self._model is None:
            local_only = self._local_model_path and os.path.isdir(self._local_model_path)
            logger.info(
                f"Loading model from {'local path' if local_only else 'HuggingFace'}: {self.model_name}"
            )
            kwargs: dict = dict(
                device=self.device,
                trust_remote_code=True,
            )
            if local_only:
                kwargs["local_files_only"] = True
            else:
                kwargs["cache_folder"] = self._cache_folder
            self._model = SentenceTransformer(self.model_name, **kwargs)
            logger.info("Model loaded successfully")
        return self._model

    def get_embedding_dim(self) -> int:
        """Get the embedding dimension"""
        return self.model.get_sentence_embedding_dimension()

    def get_max_seq_length(self) -> int:
        """Get maximum sequence length"""
        return self.model.max_seq_length

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: Optional[bool] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings for text(s).

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            normalize: Override normalization setting

        Returns:
            List of embedding vectors (each vector is a list of floats)
        """
        # Handle single text input
        if isinstance(texts, str):
            single_text = True
            texts = [texts]
        else:
            single_text = False

        # Use default normalization if not specified
        if normalize is None:
            normalize = self.normalize_embeddings

        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            convert_to_tensor=False,
        )

        # Convert to list of floats
        result = [emb.tolist() for emb in embeddings]

        # Return single embedding if input was single text
        if single_text:
            return result[0]

        return result

    def encode_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query (same as encode for single text).
        Included for API compatibility.
        """
        return self.encode(query)

    def encode_corpus(
        self,
        sentences: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> List[List[float]]:
        """
        Generate embeddings for a corpus of sentences.

        Args:
            sentences: List of sentences to embed
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar

        Returns:
            List of embedding vectors
        """
        return self.encode(
            texts=sentences,
            batch_size=batch_size,
            show_progress=show_progress,
        )

    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score
        """
        import numpy as np

        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)

        # Normalize if not already normalized
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    def unload_model(self):
        """Unload the model from memory"""
        if self._model is not None:
            del self._model
            self._model = None
            # Clear CUDA cache if using GPU
            if self.device == "cuda":
                torch.cuda.empty_cache()
            logger.info("Model unloaded from memory")
