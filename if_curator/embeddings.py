"""
Unified embedding interface for faces and objects.

- Faces: InsightFace (ArcFace/Buffalo_L) — or reuse from Immich
- Objects: SigLIP (Vision Transformer via transformers)
- Caching: Disk-based cache avoids recomputation on reruns
"""

import contextlib
import importlib
import logging
import os
import warnings

import cv2
import numpy as np
from PIL import Image

from .cache import get_cache

logger = logging.getLogger(__name__)

# Lazy-loaded singletons
_insightface_app = None
_insightface_loaded = False
_siglip_model = None
_siglip_processor = None
_siglip_loaded = False


def _is_force_cpu() -> bool:
    """Check if CPU mode is forced via environment variable."""
    return os.getenv("FORCE_CPU", "").lower() in ("true", "1", "yes")


# =============================================================================
# InsightFace (Faces)
# =============================================================================


def get_insightface_app():
    """Singleton for InsightFace app with automatic GPU/CPU fallback."""
    global _insightface_app, _insightface_loaded
    if _insightface_loaded:
        return _insightface_app
    _insightface_loaded = True

    try:
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        # Get providers, excluding TensorRT to avoid noisy errors
        providers = [p for p in ort.get_available_providers() if p != "TensorrtExecutionProvider"]
        logger.info(f"Available ONNX providers: {providers}")

        # Determine device: 0 for GPU, -1 for CPU
        gpu_providers = {
            "CUDAExecutionProvider",
            "ROCmExecutionProvider",
            "MPSExecutionProvider",
            "CoreMLExecutionProvider",
        }
        ctx_id = -1 if _is_force_cpu() else (0 if gpu_providers & set(providers) else -1)

        device_str = "GPU" if ctx_id >= 0 else "CPU"
        logger.info(f"Loading InsightFace Buffalo_L on {device_str} (ctx_id={ctx_id})...")

        # Suppress C-level output during model loading
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            _insightface_app = FaceAnalysis(name="buffalo_l", root="~/.insightface", providers=providers)
            _insightface_app.prepare(ctx_id=ctx_id, det_size=(640, 640))

        return _insightface_app

    except ImportError:
        logger.error("InsightFace not installed!")
        return None
    except Exception as e:
        logger.error(f"Failed to load InsightFace: {e}")
        # Retry on CPU if GPU failed
        if ctx_id == 0:
            logger.warning("Retrying InsightFace on CPU...")
            try:
                from insightface.app import FaceAnalysis

                _insightface_app = FaceAnalysis(name="buffalo_l", root="~/.insightface")
                _insightface_app.prepare(ctx_id=-1, det_size=(640, 640))
                return _insightface_app
            except Exception as ex:
                logger.error(f"CPU fallback failed: {ex}")
        return None


def get_face_embedding(img_pil: Image.Image) -> np.ndarray | None:
    """Get embedding of the largest face in a PIL image."""
    app = get_insightface_app()
    if not app:
        return None

    try:
        # InsightFace expects BGR cv2 image
        img_bgr = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)

        # Suppress scikit-image FutureWarning from InsightFace's face_align.py
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*estimate.*is deprecated", category=FutureWarning)
            faces = app.get(img_bgr)

        if not faces:
            return None

        # Return embedding of largest face
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return largest.embedding
    except Exception as e:
        logger.error(f"Error getting face embedding: {e}")
        return None


# =============================================================================
# SigLIP (Objects)
# =============================================================================


def get_siglip_model():
    """Singleton for SigLIP model and processor with GPU auto-detection."""
    global _siglip_model, _siglip_processor, _siglip_loaded
    if _siglip_loaded:
        return _siglip_model, _siglip_processor
    _siglip_loaded = True

    try:
        import warnings

        import torch
        from transformers import AutoImageProcessor, SiglipVisionModel

        model_name = "google/siglip-base-patch16-224"
        logger.info(f"Loading SigLIP model ({model_name})...")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", message=".*use_fast.*")
            _siglip_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
            _siglip_model = SiglipVisionModel.from_pretrained(model_name)

        _siglip_model.eval()

        # Move to GPU if available
        if not _is_force_cpu():
            if torch.cuda.is_available():
                _siglip_model = _siglip_model.cuda()
                logger.info("SigLIP running on CUDA GPU")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                _siglip_model = _siglip_model.to("mps")
                logger.info("SigLIP running on Apple MPS")
            else:
                logger.info("SigLIP running on CPU")
        else:
            logger.info("FORCE_CPU set. SigLIP running on CPU")

        return _siglip_model, _siglip_processor

    except ImportError as e:
        logger.error(f"transformers/torch not installed: {e}")
        return None, None
    except Exception as e:
        logger.error(f"Failed to load SigLIP: {e}")
        return None, None


def get_object_embedding(img_pil: Image.Image) -> np.ndarray | None:
    """Get 768-dim SigLIP embedding for an image."""
    model, processor = get_siglip_model()
    if model is None:
        return None

    try:
        import torch

        inputs = processor(images=img_pil, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            return outputs.pooler_output.squeeze().cpu().numpy()
    except Exception as e:
        logger.error(f"Error getting object embedding: {e}")
        return None


def get_object_embeddings_batch(images: list[Image.Image]) -> list[np.ndarray | None]:
    """Get SigLIP embeddings for a batch of images (GPU-efficient)."""
    model, processor = get_siglip_model()
    if model is None:
        return [None] * len(images)

    try:
        import torch

        inputs = processor(images=images, return_tensors="pt", padding=True)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.pooler_output.cpu().numpy()
            return [embeddings[i] for i in range(len(embeddings))]
    except Exception as e:
        logger.error(f"Error in batch embedding: {e}")
        # Fall back to individual computation
        return [get_object_embedding(img) for img in images]


# =============================================================================
# Unified Interface with Caching
# =============================================================================


def get_embedding(
    img_pil: Image.Image,
    entity_type: str = "face",
    asset_id: str | None = None,
    immich_embedding: np.ndarray | None = None,
) -> np.ndarray | None:
    """Get embedding for an image based on entity type.

    Priority:
    1. Pre-fetched Immich embedding (if provided)
    2. Disk cache (if enabled and asset_id provided)
    3. Local model computation (InsightFace or SigLIP)

    Args:
        img_pil: The image to embed
        entity_type: 'face' or 'object'
        asset_id: Optional asset ID for cache lookup
        immich_embedding: Optional pre-fetched embedding from Immich API
    """
    from .config import Config

    use_cache = Config.ENABLE_CACHE and asset_id is not None
    cache = get_cache(Config.CACHE_DIR) if use_cache else None
    model_key = "immich" if entity_type == "face" else "siglip"

    # 1. Use Immich embedding if provided
    if immich_embedding is not None:
        if cache:
            cache.put(asset_id, immich_embedding, model_key)
        return immich_embedding

    # 2. Check disk cache
    if cache:
        cached = cache.get(asset_id, model_key)
        if cached is not None:
            return cached

    # 3. Compute locally
    if entity_type == "face":
        emb = get_face_embedding(img_pil)
        model_key = "insightface"
    else:
        emb = get_object_embedding(img_pil)

    # Cache the result
    if emb is not None and cache:
        cache.put(asset_id, emb, model_key)

    return emb


def _is_module_available(module_name: str) -> bool:
    """Check if a Python module is importable without importing it fully."""
    try:
        importlib.util.find_spec(module_name)
        return True
    except (ModuleNotFoundError, ValueError):
        return False


def is_embedding_available(entity_type: str = "face", *, load: bool = False) -> bool:
    """Check if embedding model is available for the given entity type.

    By default this performs a lightweight import-check only (no model loading).
    Pass ``load=True`` to actually load the model (expensive, hundreds of MB).

    Args:
        entity_type: 'face' or 'object'
        load: If True, fully load the model to verify. If False (default),
              only check that the required packages are importable.
    """
    if load:
        if entity_type == "face":
            return get_insightface_app() is not None
        model, _ = get_siglip_model()
        return model is not None

    # Lightweight check: just verify the packages are importable
    if entity_type == "face":
        return _is_module_available("insightface") and _is_module_available("onnxruntime")
    return _is_module_available("transformers") and _is_module_available("torch")


def load_embedding_model(entity_type: str = "face") -> bool:
    """Explicitly load the embedding model for the given entity type.

    Returns True if the model loaded successfully.
    """
    if entity_type == "face":
        return get_insightface_app() is not None
    model, _ = get_siglip_model()
    return model is not None

