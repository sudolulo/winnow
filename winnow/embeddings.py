"""
Embedding interface for face diversity selection.

- Faces: InsightFace (ArcFace/Buffalo_L) — or reuse from Immich
- Caching: Disk-based cache avoids recomputation on reruns
"""

import importlib
import logging
import os
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .cache import get_cache
from .config import _getenv_bool

logger = logging.getLogger(__name__)


@contextmanager
def _suppress_output():
    """Suppress stdout/stderr at the file-descriptor level, silencing C extension noise."""
    devnull_fd = None
    saved_out = None
    saved_err = None
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved_out = os.dup(1)
        saved_err = os.dup(2)
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        # Each block is a separate sequential statement. A BaseException (e.g.
        # KeyboardInterrupt) raised inside block N would propagate past blocks N+1
        # and N+2, leaving saved_err or devnull_fd unclosed. In CPython, KI is
        # delivered between bytecodes, not mid-syscall; os.dup2 is a single C call
        # and completes atomically, so this race is not realistically triggerable.
        if saved_out is not None:
            try:
                os.dup2(saved_out, 1)
            except OSError:
                pass
            finally:
                try:
                    os.close(saved_out)
                except OSError:
                    pass
        if saved_err is not None:
            try:
                os.dup2(saved_err, 2)
            except OSError:
                pass
            finally:
                try:
                    os.close(saved_err)
                except OSError:
                    pass
        if devnull_fd is not None:
            try:
                os.close(devnull_fd)
            except OSError:
                pass


# Lazy-loaded singleton
_insightface_app = None
_insightface_loaded = False


def _is_force_cpu() -> bool:
    """Check if CPU mode is forced via environment variable."""
    return _getenv_bool("FORCE_CPU", False)


def _preload_cuda_libs() -> None:
    """Preload CUDA/cuDNN DLLs so onnxruntime-gpu registers CUDAExecutionProvider.

    Starting with onnxruntime-gpu 1.19+, CUDA/cuDNN libraries are no longer
    bundled inside the ORT package — they come from the nvidia-* pip packages.
    preload_dlls() locates them automatically via site-packages discovery.
    """
    try:
        import onnxruntime
        if hasattr(onnxruntime, "preload_dlls"):
            onnxruntime.preload_dlls(cuda=True, cudnn=True)
            logger.debug("Preloaded CUDA/cuDNN DLLs for onnxruntime-gpu")
        else:
            logger.debug("onnxruntime.preload_dlls() not available (ORT < 1.21)")
    except Exception as e:
        logger.warning("Failed to preload CUDA/cuDNN DLLs: %s", e)


# =============================================================================
# InsightFace (Faces)
# =============================================================================


def get_insightface_app():
    """Singleton for InsightFace app with automatic GPU/CPU fallback."""
    global _insightface_app, _insightface_loaded
    if _insightface_loaded:
        return _insightface_app
    _insightface_loaded = True

    ctx_id = -1
    insightface_home = os.environ.get("INSIGHTFACE_HOME", os.path.expanduser("~/.insightface"))
    try:
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        # Preload CUDA/cuDNN DLLs before any ORT InferenceSession is created.
        # Silently no-ops on ROCm/Intel builds where preload_dlls() is absent.
        _preload_cuda_libs()

        # Disk cache check — lets the user know whether a download is coming
        buffalo_path = Path(insightface_home) / "models" / "buffalo_l"
        if buffalo_path.exists() and any(buffalo_path.iterdir()):
            logger.info("InsightFace Buffalo_L: found in model cache")
        else:
            logger.info("InsightFace Buffalo_L: not cached — downloading now (~300 MB)")

        # Get providers, excluding TensorRT to avoid noisy errors
        providers = [p for p in ort.get_available_providers() if p != "TensorrtExecutionProvider"]
        logger.debug("ONNX providers available: %s", providers)

        gpu_providers = {
            "CUDAExecutionProvider",
            "ROCmExecutionProvider",
            "MPSExecutionProvider",
            "CoreMLExecutionProvider",
            "OpenVINOExecutionProvider",
        }
        has_gpu_provider = bool(gpu_providers & set(providers))
        ctx_id = -1 if _is_force_cpu() else (0 if has_gpu_provider else -1)

        # For OpenVINO EP, inject device_type from env var (default CPU; set GPU for Intel Arc/iGPU)
        has_openvino = "OpenVINOExecutionProvider" in providers
        if has_openvino:
            openvino_device = os.getenv("OPENVINO_DEVICE", "CPU")
            providers = [
                ("OpenVINOExecutionProvider", {"device_type": openvino_device})
                if p == "OpenVINOExecutionProvider" else p
                for p in providers
            ]
            logger.debug("OpenVINO EP: device_type=%s", openvino_device)

        if not has_gpu_provider and not _is_force_cpu():
            logger.warning(
                "No GPU execution provider found — running InsightFace on CPU. "
                "Ensure the container has GPU access and the correct variant image is used "
                "(gpu for NVIDIA, rocm for AMD, intel for Intel Arc/iGPU)."
            )

        if ctx_id < 0:
            device_str = "CPU"
        elif has_openvino:
            device_str = f"OpenVINO ({os.getenv('OPENVINO_DEVICE', 'CPU')})"
        else:
            device_str = "GPU"
        logger.info("InsightFace Buffalo_L: loading into memory on %s...", device_str)

        t0 = time.time()
        with _suppress_output():
            _insightface_app = FaceAnalysis(name="buffalo_l", root=insightface_home, providers=providers)
            _insightface_app.prepare(ctx_id=ctx_id, det_size=(640, 640))

        logger.info("InsightFace Buffalo_L: ready on %s (%.1fs)", device_str, time.time() - t0)
        return _insightface_app

    except ImportError:
        logger.error("InsightFace not installed!")
        return None
    except Exception as e:
        logger.error("Failed to load InsightFace: %s", e)
        if ctx_id == 0:
            logger.warning("InsightFace GPU load failed — retrying on CPU...")
            try:
                from insightface.app import FaceAnalysis

                t0 = time.time()
                with _suppress_output():
                    _insightface_app = FaceAnalysis(
                        name="buffalo_l",
                        root=insightface_home,
                        providers=["CPUExecutionProvider"],
                    )
                    _insightface_app.prepare(ctx_id=-1, det_size=(640, 640))
                logger.info("InsightFace Buffalo_L: ready on CPU (fallback, %.1fs)", time.time() - t0)
                return _insightface_app
            except Exception as ex:
                logger.error("InsightFace CPU fallback failed: %s", ex)
        return None


def get_face_embedding(img_pil: Image.Image) -> np.ndarray | None:
    """Get embedding of the largest face in a PIL image."""
    app = get_insightface_app()
    if not app:
        return None

    try:
        # InsightFace expects BGR cv2 image; normalise mode first so RGBA/grayscale don't
        # raise a channel-count error inside cvtColor.
        img_bgr = cv2.cvtColor(np.asarray(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

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
        logger.error("Error getting face embedding: %s", e)
        return None


# =============================================================================
# Embedding Interface with Caching
# =============================================================================


def get_embedding(
    img_pil: Image.Image,
    asset_id: str | None = None,
) -> np.ndarray | None:
    """Get embedding for a face image.

    Checks disk cache first (if enabled and asset_id provided),
    then falls back to local InsightFace computation.
    """
    from .config import Config

    use_cache = Config.ENABLE_CACHE and asset_id is not None
    cache = get_cache(Config.DATA_DIR) if use_cache else None

    if cache:
        cached = cache.get(asset_id, "insightface")
        if cached is not None:
            return cached

    emb = get_face_embedding(img_pil)

    if emb is not None and cache:
        cache.put(asset_id, emb, "insightface")

    return emb


def _is_module_available(module_name: str) -> bool:
    """Check if a Python module is importable without importing it fully."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


def is_embedding_available(*, load: bool = False) -> bool:
    """Check if InsightFace is available.

    By default this performs a lightweight import-check only (no model loading).
    Pass ``load=True`` to actually load the model (expensive, ~300 MB).
    """
    if load:
        return get_insightface_app() is not None
    return _is_module_available("insightface") and _is_module_available("onnxruntime")


def load_embedding_model() -> bool:
    """Explicitly load InsightFace. Returns True if the model loaded successfully."""
    return get_insightface_app() is not None

