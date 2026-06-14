"""Disk-based embedding cache.

Caches embeddings keyed by (asset_id, model_version) to avoid
recomputing on reruns. Uses numpy binary format for fast I/O.
"""

import hashlib
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MODEL_VERSIONS = {
    "immich": "immich_buffalo_l_v1",
}


def _insightface_model_fingerprint() -> str:
    """Derive a version string from buffalo_l .onnx file sizes and mtimes.

    Changes automatically when model files are replaced or updated, preventing
    stale embeddings from a previous model being served from cache.
    Falls back to a static string before the model is downloaded (first run).
    """
    insightface_home = os.environ.get("INSIGHTFACE_HOME", os.path.expanduser("~/.insightface"))
    model_dir = Path(insightface_home) / "models" / "buffalo_l"
    if not model_dir.exists():
        return "buffalo_l_v1"
    onnx_files = sorted(model_dir.glob("*.onnx"))
    if not onnx_files:
        return "buffalo_l_v1"
    fingerprint = "|".join(
        f"{f.name}:{f.stat().st_size}:{int(f.stat().st_mtime)}"
        for f in onnx_files
    )
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:12]


class EmbeddingCache:
    """Simple disk-based embedding cache.

    Embeddings are stored as .npy files in a flat directory,
    keyed by a hash of (asset_id, model_version). The InsightFace version
    is derived from buffalo_l model file metadata so the cache auto-invalidates
    when model files are replaced or updated.
    """

    def __init__(self, cache_dir: str = ".if_cache") -> None:
        self.cache_dir = cache_dir
        self._ensured = False
        self._model_versions = {
            **MODEL_VERSIONS,
            "insightface": _insightface_model_fingerprint(),
        }

    def _ensure_dir(self) -> None:
        if not self._ensured:
            os.makedirs(self.cache_dir, exist_ok=True)
            self._ensured = True

    def _key(self, asset_id: str, model: str) -> str:
        version = self._model_versions.get(model, model)
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
            logger.debug("Cache write failed for %s: %s", asset_id, e)

    def clear(self) -> None:
        """Delete all cached embeddings."""
        if not os.path.isdir(self.cache_dir):
            return
        count = 0
        for f in os.listdir(self.cache_dir):
            if f.endswith(".npy"):
                os.remove(os.path.join(self.cache_dir, f))
                count += 1
        logger.info("Cleared %s cached embeddings.", count)


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
