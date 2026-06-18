"""Image processing functions for cropping faces."""

import logging
import os
import warnings

import numpy as np
from PIL import Image

from .config import Config

logger = logging.getLogger(__name__)


def _save_jpeg(img: Image.Image, path: str) -> None:
    if img.mode != "RGB":
        img = img.convert("RGB")
    tmp = path + ".tmp"
    try:
        img.save(tmp, format="JPEG")
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def align_face(img: Image.Image, landmarks: list[list[float]] | np.ndarray) -> Image.Image | None:
    """Align face using 5-point landmarks to standard ArcFace input format (112x112).

    This produces a normalized face crop that matches exactly what ArcFace
    was trained on, improving recognition accuracy.

    Args:
        img: Full PIL image containing the face
        landmarks: 5-point facial landmarks [[x,y], ...] (eyes, nose, mouth corners)

    Returns:
        Aligned 112x112 face image, or None if alignment fails
    """
    try:
        from insightface.utils.face_align import norm_crop

        img_np = np.asarray(img)
        lm = np.array(landmarks, dtype=np.float32)
        if lm.shape != (5, 2):
            logger.debug("Invalid landmark shape: %s, expected (5, 2)", lm.shape)
            return None
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*estimate.*is deprecated", category=FutureWarning)
            aligned = norm_crop(img_np, lm)
        return Image.fromarray(aligned)
    except ImportError:
        logger.debug("InsightFace not available for face alignment")
        return None
    except Exception as e:
        logger.debug("Face alignment failed: %s", e)
        return None


def process_face_mode(
    img: Image.Image,
    asset: dict,
    person: dict,
    output_dir: str,
    count: int,
    min_width: int | None = None,
    insightface_app=None,
) -> tuple[int, int] | str:
    """Crop face based on Immich metadata and save to output directory.

    Returns (width, height) of the saved crop, or a skip-reason string if the
    face was filtered out.
    When insightface_app is provided and ENABLE_FACE_ALIGNMENT is True,
    re-detects the face in the Immich bbox region using InsightFace to get
    precise landmarks for a proper 112x112 aligned crop. Falls back to
    bounding box crop with configurable margin if alignment is unavailable.
    """
    min_width = min_width or Config.MIN_FACE_WIDTH

    # Find face metadata for this person
    face_info = None
    people_list = asset.get("people") or []
    for p in people_list:
        if p is None:
            continue
        if p.get("id") == person["id"] and (faces := p.get("faces")):
            face_info = faces[0]
            break

    if not face_info:
        logger.debug("No face info for %s in asset %s", person.get("name"), asset.get("id"))
        return "no face metadata"

    img_w, img_h = img.size
    meta_w = face_info.get("imageWidth") or 0
    meta_h = face_info.get("imageHeight") or 0

    # Scale bounding box from detection-image space to actual image dimensions.
    # Fall back to 1.0 if Immich omits the field — bbox is assumed to already
    # be in image space (correct for thumbnails, wrong for full-res).
    scale_x = img_w / meta_w if meta_w else 1.0
    scale_y = img_h / meta_h if meta_h else 1.0
    x1 = face_info["boundingBoxX1"] * scale_x
    y1 = face_info["boundingBoxY1"] * scale_y
    x2 = face_info["boundingBoxX2"] * scale_x
    y2 = face_info["boundingBoxY2"] * scale_y

    face_w, face_h = x2 - x1, y2 - y1
    if face_w < min_width or face_h < min_width:
        logger.debug("Face too small (%.1fx%.1f)", face_w, face_h)
        return f"face too small ({face_w:.0f}x{face_h:.0f}px, min {min_width}px)"

    # Re-detect face with InsightFace for landmark-based alignment.
    # Immich's /api/faces endpoint does not include landmarks, so the
    # align_face fallback below never fires without this step.
    if insightface_app is not None and Config.ENABLE_FACE_ALIGNMENT:
        try:
            # Expand the Immich bbox by 50% to give InsightFace enough context
            # for detection and alignment, then search for the face nearest the
            # centre of that region (handles group photos at the boundary).
            pad_x, pad_y = face_w * 0.5, face_h * 0.5
            search_box = (
                max(0, x1 - pad_x),
                max(0, y1 - pad_y),
                min(img_w, x2 + pad_x),
                min(img_h, y2 + pad_y),
            )
            search_crop = img.crop(search_box)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*estimate.*is deprecated", category=FutureWarning)
                detected = insightface_app.get(np.asarray(search_crop))
            if detected:
                cx, cy = search_crop.width / 2, search_crop.height / 2
                best = min(
                    detected,
                    key=lambda f: abs((f.bbox[0] + f.bbox[2]) / 2 - cx)
                    + abs((f.bbox[1] + f.bbox[3]) / 2 - cy),
                )
                kps = getattr(best, "kps", None)
                if kps is not None and np.asarray(kps).shape == (5, 2):
                    aligned = align_face(search_crop, kps)
                    if aligned is not None:
                        _save_jpeg(aligned, os.path.join(output_dir, f"{count}.jpg"))
                        return aligned.size
        except Exception as e:
            logger.debug("InsightFace re-detection failed for %s: %s", asset.get("id"), e)

    # Landmark alignment from Immich metadata (Immich does not currently
    # expose landmarks, so this path is a future-proofing fallback)
    if Config.ENABLE_FACE_ALIGNMENT:
        landmarks = face_info.get("landmarks") or face_info.get("landmark")
        if landmarks:
            scaled_landmarks = [[lm[0] * scale_x, lm[1] * scale_y] for lm in landmarks]
            aligned = align_face(img, scaled_landmarks)
            if aligned is not None:
                _save_jpeg(aligned, os.path.join(output_dir, f"{count}.jpg"))
                return aligned.size

    # Final fallback: bounding box crop with configurable margin
    margin = Config.FACE_MARGIN
    margin_x, margin_y = face_w * margin, face_h * margin
    crop_box = (
        max(0, x1 - margin_x),
        max(0, y1 - margin_y),
        min(img_w, x2 + margin_x),
        min(img_h, y2 + margin_y),
    )

    face_crop = img.crop(crop_box)
    _save_jpeg(face_crop, os.path.join(output_dir, f"{count}.jpg"))
    return face_crop.size



