"""
Text embedder using sentence-transformers for LLM-based embeddings.

This module provides text embedding capabilities for column names and
category values using pre-trained sentence transformers.
"""

from typing import Dict, List, Optional

import torch
from sentence_transformers import SentenceTransformer


# Default model - MiniLM is fast and produces good embeddings
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # MiniLM-L6-v2 embedding dimension


class TextEmbedder:
    """
    Embed text using sentence-transformers with caching.

    Uses a pre-trained sentence transformer to convert text strings
    (column names, category values) into dense vector embeddings.
    Embeddings are cached for efficiency.

    Args:
        model_name: HuggingFace model name. Defaults to MiniLM-L6-v2.
        device: Device to run embeddings on. Defaults to auto-detect.
        cache_size: Max number of embeddings to cache. Defaults to 100000.
        rotate: If True, evict oldest entries when cache is full. If False, cache grows unbounded.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        cache_size: int = 500000,
        rotate: bool = False,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Optional[SentenceTransformer] = None
        self._cache: Dict[str, torch.Tensor] = {}
        self._cache_size = cache_size
        self._rotate = rotate

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return EMBEDDING_DIM

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed(self, texts: List[str], show_progress: bool = False) -> torch.Tensor:
        """
        Embed a list of texts.

        Args:
            texts: List of text strings to embed.
            show_progress: Whether to show progress bar for batch encoding.

        Returns:
            Tensor of shape (len(texts), embedding_dim) with embeddings.
        """
        if not texts:
            return torch.empty(0, self.embedding_dim, device=self.device)

        text_strs = [str(t) for t in texts]

        # Fast path: check if all texts are cached
        all_cached = all(t in self._cache for t in text_strs)
        if all_cached:
            return torch.stack([self._cache[t] for t in text_strs])

        # Separate cached vs missing, and save cached embeddings
        missing_texts = []
        cached_embeddings = {}  # text -> embedding for texts in this batch
        for text_str in text_strs:
            if text_str in self._cache:
                cached_embeddings[text_str] = self._cache[text_str]
            elif text_str not in cached_embeddings:  # not already queued
                missing_texts.append(text_str)

        # Compute missing embeddings
        if missing_texts:
            with torch.no_grad():
                new_embeddings = self.model.encode(
                    missing_texts,
                    convert_to_tensor=True,
                    show_progress_bar=show_progress,
                    device=self.device,
                )
                # Convert to float32 for consistency
                new_embeddings = new_embeddings.float()

            # Cache new embeddings and add to cached_embeddings
            for text, embedding in zip(missing_texts, new_embeddings):
                # FIFO cache eviction only when rotate is enabled
                if self._rotate and len(self._cache) >= self._cache_size:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                self._cache[text] = embedding
                cached_embeddings[text] = embedding

        # Build result from cached_embeddings (safe from eviction)
        return torch.stack([cached_embeddings[t] for t in text_strs])

    def embed_batch_no_cache(self, texts: List[str]) -> torch.Tensor:
        """
        Embed texts directly without caching. Faster for large batches with many unique values.

        Args:
            texts: List of text strings to embed.

        Returns:
            Tensor of shape (len(texts), embedding_dim) with embeddings.
        """
        if not texts:
            return torch.empty(0, self.embedding_dim, device=self.device)

        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=False,
                device=self.device,
            )
            return embeddings.float()

    def embed_single(self, text: str) -> torch.Tensor:
        """
        Embed a single text string.

        Args:
            text: Text string to embed.

        Returns:
            Tensor of shape (embedding_dim,) with embedding.
        """
        return self.embed([text])[0]

    def precompute(self, texts: List[str], show_progress: bool = True) -> None:
        """
        Pre-compute and cache embeddings for a list of texts.

        Useful for pre-computing all unique values during fit().

        Args:
            texts: List of text strings to pre-compute.
            show_progress: Whether to show progress bar.
        """
        # Filter to texts not already cached
        new_texts = [str(t) for t in texts if str(t) not in self._cache]
        if new_texts:
            self.embed(new_texts, show_progress=show_progress)

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()

    def get_cached_embedding(self, text: str) -> Optional[torch.Tensor]:
        """
        Get embedding from cache without computing.

        Args:
            text: Text string to look up.

        Returns:
            Embedding tensor if cached, None otherwise.
        """
        return self._cache.get(str(text))

    def to(self, device: str) -> "TextEmbedder":
        """
        Move embedder and cached embeddings to device.

        Args:
            device: Target device ('cuda' or 'cpu').

        Returns:
            Self for method chaining.
        """
        self.device = device
        if self._model is not None:
            self._model = self._model.to(device)
        # Move cached embeddings
        self._cache = {k: v.to(device) for k, v in self._cache.items()}
        return self

    def state_dict(self) -> Dict:
        """
        Get state for serialization.

        Returns:
            Dict with model name (cache not saved to reduce checkpoint size).
        """
        return {
            "model_name": self.model_name,
        }

    def load_state_dict(self, state: Dict) -> None:
        """
        Load state from serialization.

        Args:
            state: State dict from state_dict().
        """
        self.model_name = state.get("model_name", DEFAULT_MODEL)
        # Cache is not loaded from checkpoints (rebuilt on demand)
