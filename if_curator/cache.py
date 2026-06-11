"""Disk-based embedding cache.

Caches embeddings keyed by (asset_id, model_version) to avoid
recomputing on reruns. Uses numpy binary format for fast I/O.
"""

import hashlib
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

# Model versions — bump these when the upstream model changes
MODEL_VERSIONS = {
    "insightface": "buffalo_l_v1",
    "siglip": "siglip-base-patch16-224_v1",
    "immich": "immich_buffalo_l_v1",
}


class EmbeddingCache:
    """Simple disk-based embedding cache.

    Embeddings are stored as .npy files in a flat directory,
    keyed by a hash of (asset_id, model_version).
    """

    def __init__(self, cache_dir: str = ".if_cache") -> None:
        self.cache_dir = cache_dir
        self._ensured = False

    def _ensure_dir(self) -> None:
        if not self._ensured:
            os.makedirs(self.cache_dir, exist_ok=True)
            self._ensured = True

    @staticmethod
    def _key(asset_id: str, model: str) -> str:
        version = MODEL_VERSIONS.get(model, model)
        raw = f"{asset_id}:{version}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _path(self, asset_id: str, model: str) -> str:
        return os.path.join(self.cache_dir, f"{self._key(asset_id, model)}.npy")

    def get(self, asset_id: str, model: str = "insightface") -> np.ndarray | None:
        """Retrieve cached embedding, or None if not cached."""
        path = self._path(asset_id, model)
        if os.path.exists(path):
            try:
                return np.load(path)
            except Exception:
                return None
        return None

    def put(self, asset_id: str, embedding: np.ndarray, model: str = "insightface") -> None:
        """Store an embedding in the cache."""
        self._ensure_dir()
        try:
            np.save(self._path(asset_id, model), embedding)
        except Exception as e:
            logger.debug(f"Cache write failed for {asset_id}: {e}")

    def clear(self) -> None:
        """Delete all cached embeddings."""
        if not os.path.isdir(self.cache_dir):
            return
        count = 0
        for f in os.listdir(self.cache_dir):
            if f.endswith(".npy"):
                os.remove(os.path.join(self.cache_dir, f))
                count += 1
        logger.info(f"Cleared {count} cached embeddings.")


# Singleton instance
_cache: EmbeddingCache | None = None


def get_cache(cache_dir: str = ".if_cache") -> EmbeddingCache:
    """Get or create the singleton cache instance.

    Note: The ``cache_dir`` parameter is only used when creating the
    singleton for the first time. Subsequent calls return the existing
    instance regardless of ``cache_dir``. If you need a cache with a
    different directory, instantiate ``EmbeddingCache`` directly.
    """
    global _cache
    if _cache is None:
        _cache = EmbeddingCache(cache_dir)
    return _cache

