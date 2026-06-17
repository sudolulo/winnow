"""Image quality assessment for training data curation.

Filters out images that would hurt Frigate's ArcFace model training:
blur, grayscale/IR, over/underexposure, tiny faces, low-confidence detections.
"""

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _laplacian_var(img_np: np.ndarray) -> float:
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if img_np.ndim == 3 else img_np
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


@dataclass
class QualityResult:
    """Result of quality assessment on a face/image crop."""

    passed: bool
    reasons: list[str] = field(default_factory=list)
    blur_score: float | None = None

    @property
    def reason(self) -> str:
        return "; ".join(self.reasons) if self.reasons else "OK"


def check_blur(img_np: np.ndarray, threshold: float = 100.0) -> tuple[bool, str]:
    """Detect blur using Laplacian variance.

    Lower variance = blurrier image. ArcFace needs clear facial features.
    """
    variance = _laplacian_var(img_np)
    if variance < threshold:
        return False, f"Blurry (laplacian={variance:.1f}, threshold={threshold})"
    return True, ""


def check_grayscale(img_np: np.ndarray) -> tuple[bool, str]:
    """Detect grayscale/IR images by checking channel similarity.

    ArcFace is trained on color images; IR/grayscale degrades recognition.
    """
    if img_np.ndim != 3 or img_np.shape[2] < 3:
        return False, "Grayscale (single channel)"

    # Compare channel means — IR/grayscale has nearly identical R, G, B
    means = img_np[:, :, :3].mean(axis=(0, 1))
    max_diff = max(abs(means[0] - means[1]), abs(means[1] - means[2]), abs(means[0] - means[2]))

    if max_diff < 5.0:
        return False, f"Grayscale/IR (channel diff={max_diff:.1f})"
    return True, ""


def check_exposure(img_np: np.ndarray, lo: float = 30.0, hi: float = 225.0) -> tuple[bool, str]:
    """Check for severe under/overexposure using mean brightness.

    Extremely dark or blown-out faces lack usable features.
    """
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if img_np.ndim == 3 else img_np
    mean_brightness = gray.mean()

    if mean_brightness < lo:
        return False, f"Underexposed (brightness={mean_brightness:.1f}, min={lo})"
    if mean_brightness > hi:
        return False, f"Overexposed (brightness={mean_brightness:.1f}, max={hi})"
    return True, ""


def check_face_size(face_width: float, face_height: float, min_px: int = 50) -> tuple[bool, str]:
    """Validate face crop is large enough for meaningful features."""
    if face_width < min_px or face_height < min_px:
        return False, f"Face too small ({face_width:.0f}x{face_height:.0f}, min={min_px}px)"
    return True, ""


def check_confidence(score: float | None, min_conf: float = 0.7) -> tuple[bool, str]:
    """Check detection confidence from Immich.

    Low confidence often means partial, occluded, or false-positive faces.
    """
    if score is None:
        return True, ""  # No score available, pass by default
    if score < min_conf:
        return False, f"Low confidence ({score:.2f}, min={min_conf})"
    return True, ""


def assess_quality(
    img: Image.Image,
    face_bbox: tuple[float, float, float, float] | None = None,
    confidence: float | None = None,
    blur_threshold: float = 100.0,
    min_face_px: int = 50,
    min_confidence: float = 0.7,
) -> QualityResult:
    """Run all quality checks on an image.

    Args:
        img: PIL Image (RGB)
        face_bbox: (x1, y1, x2, y2) bounding box, or None to skip face-size check
        confidence: Detection confidence score from Immich, or None
        blur_threshold: Laplacian variance threshold for blur detection
        min_face_px: Minimum face dimension in pixels
        min_confidence: Minimum acceptable detection confidence

    Returns:
        QualityResult with passed=True if all checks pass
    """
    img_np = np.asarray(img)
    reasons = []

    # Compute laplacian variance once (used by check_blur and stored as blur_score)
    blur_score = _laplacian_var(img_np)

    checks = [
        (
            blur_score >= blur_threshold,
            f"Blurry (laplacian={blur_score:.1f}, threshold={blur_threshold})" if blur_score < blur_threshold else "",
        ),
        check_grayscale(img_np),
        check_exposure(img_np),
        check_confidence(confidence, min_confidence),
    ]

    if face_bbox is not None:
        x1, y1, x2, y2 = face_bbox
        checks.append(check_face_size(x2 - x1, y2 - y1, min_face_px))

    for passed, reason in checks:
        if not passed:
            reasons.append(reason)

    return QualityResult(passed=len(reasons) == 0, reasons=reasons, blur_score=blur_score)


def blur_score_from_image(img: Image.Image, max_dim: int = 1440) -> float | None:
    """Compute Laplacian-variance blur score, capped at max_dim px to normalise scale.

    Caps resolution so full-res and thumbnail scores are comparable — Laplacian
    variance grows with pixel count, making uncapped full-res scores much larger
    than thumbnail scores for the same perceived sharpness.

    Returns None on error so callers can distinguish a failed measurement from a
    legitimately low (near-zero) score.
    """
    try:
        score_img = img.convert("RGB") if img.mode != "RGB" else img
        if score_img.width > max_dim or score_img.height > max_dim:
            if score_img is img:
                score_img = score_img.copy()
            score_img.thumbnail((max_dim, max_dim), Image.LANCZOS)
        return _laplacian_var(np.array(score_img))
    except Exception as exc:
        logger.debug("blur_score_from_image failed: %s", exc)
        return None

