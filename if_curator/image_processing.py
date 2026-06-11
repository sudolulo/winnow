"""Image processing functions for cropping faces and objects."""

import logging
import os

import numpy as np
from PIL import Image

from .config import Config

logger = logging.getLogger(__name__)

# Lazy singleton
_yolo_model = None


def _save_jpeg(img: Image.Image, path: str) -> None:
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(path, format="JPEG")


def get_yolo_model():
    """Singleton for YOLO model."""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO

        logger.info("Loading YOLOv9c model...")
        _yolo_model = YOLO("yolov9c.pt")
    return _yolo_model


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
            logger.debug(f"Invalid landmark shape: {lm.shape}, expected (5, 2)")
            return None
        aligned = norm_crop(img_np, lm)
        return Image.fromarray(aligned)
    except ImportError:
        logger.debug("InsightFace not available for face alignment")
        return None
    except Exception as e:
        logger.debug(f"Face alignment failed: {e}")
        return None


def process_face_mode(
    img: Image.Image,
    asset: dict,
    person: dict,
    output_dir: str,
    count: int,
    min_width: int | None = None,
) -> bool:
    """Crop face based on Immich metadata and save to output directory.

    If face alignment is enabled and landmarks are available, produces
    an aligned 112x112 crop. Otherwise falls back to bounding box crop
    with configurable margin.
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
        logger.debug(f"No face info for {person.get('name')} in asset {asset.get('id')}")
        return False

    img_w, img_h = img.size
    meta_w = face_info.get("imageWidth") or img_w
    meta_h = face_info.get("imageHeight") or img_h

    # Scale bounding box to actual image dimensions
    scale_x, scale_y = img_w / meta_w, img_h / meta_h
    x1 = face_info["boundingBoxX1"] * scale_x
    y1 = face_info["boundingBoxY1"] * scale_y
    x2 = face_info["boundingBoxX2"] * scale_x
    y2 = face_info["boundingBoxY2"] * scale_y

    face_w, face_h = x2 - x1, y2 - y1
    if face_w < min_width or face_h < min_width:
        logger.debug(f"Face too small ({face_w:.1f}x{face_h:.1f})")
        return False

    # Try face alignment if enabled and landmarks available
    if Config.ENABLE_FACE_ALIGNMENT:
        landmarks = face_info.get("landmarks") or face_info.get("landmark")
        if landmarks:
            # Scale landmarks
            scaled_landmarks = [[lm[0] * scale_x, lm[1] * scale_y] for lm in landmarks]
            aligned = align_face(img, scaled_landmarks)
            if aligned is not None:
                _save_jpeg(aligned, os.path.join(output_dir, f"{count}.jpg"))
                return True

    # Fall back to bounding box crop with configurable margin
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
    return True


def process_object_mode(
    img: Image.Image,
    config: dict,
    output_dir: str,
    count: int,
) -> bool:
    """Detect and crop objects using YOLO."""
    try:
        model = get_yolo_model()
        target_class = config.get("object_class", "dog")
        device = "cpu" if os.getenv("FORCE_CPU", "").lower() in ("true", "1", "yes") else None

        results = model(img, verbose=False, device=device)

        found = False
        class_idx = 0  # Sequential counter per target class (Issue #10)
        for box in (box for r in results for box in r.boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if 0 <= cls_id < len(model.names) and model.names[cls_id] == target_class and conf > 0.5:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                _save_jpeg(
                    img.crop((x1, y1, x2, y2)),
                    os.path.join(output_dir, f"{count}_{class_idx}.jpg"),
                )
                class_idx += 1
                found = True

        return found
    except Exception as e:
        logger.error(f"YOLO processing failed: {e}")
        return False


def process_full_mode(img: Image.Image, output_dir: str, count: int) -> bool:
    """Save full image."""
    _save_jpeg(img, os.path.join(output_dir, f"{count}.jpg"))
    return True

