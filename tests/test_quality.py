"""Tests for image quality filtering functions."""

import numpy as np
from PIL import Image


def _rgb_image(r, g, b, size=(100, 100)) -> Image.Image:
    arr = np.full((*size, 3), [r, g, b], dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _noisy_color_image(size=(100, 100)) -> Image.Image:
    """Noisy image with a strong red channel so grayscale check passes."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (*size, 3), dtype=np.uint8)
    arr[:, :, 0] = np.clip(arr[:, :, 0].astype(int) + 80, 0, 255).astype(np.uint8)
    arr[:, :, 2] = np.clip(arr[:, :, 2].astype(int) - 80, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


# ── check_blur ────────────────────────────────────────────────────────────────

def test_blur_rejects_flat_image():
    from winnow.quality import check_blur
    flat = np.full((100, 100, 3), 128, dtype=np.uint8)
    passed, reason = check_blur(flat, threshold=100.0)
    assert not passed
    assert "Blurry" in reason


def test_blur_passes_noisy_color_image():
    from winnow.quality import check_blur
    img = _noisy_color_image()
    passed, _ = check_blur(np.asarray(img), threshold=100.0)
    assert passed


# ── check_grayscale ───────────────────────────────────────────────────────────

def test_grayscale_rejects_ir_image():
    from winnow.quality import check_grayscale
    gray = np.full((100, 100, 3), 128, dtype=np.uint8)
    passed, reason = check_grayscale(gray)
    assert not passed
    assert "Grayscale" in reason


def test_grayscale_passes_color_image():
    from winnow.quality import check_grayscale
    color = np.zeros((100, 100, 3), dtype=np.uint8)
    color[:, :, 0] = 200  # strong red channel
    passed, _ = check_grayscale(color)
    assert passed


def test_grayscale_rejects_single_channel():
    from winnow.quality import check_grayscale
    single = np.full((100, 100, 1), 128, dtype=np.uint8)
    passed, reason = check_grayscale(single)
    assert not passed


# ── check_exposure ────────────────────────────────────────────────────────────

def test_exposure_rejects_black_image():
    from winnow.quality import check_exposure
    black = np.zeros((100, 100, 3), dtype=np.uint8)
    passed, reason = check_exposure(black)
    assert not passed
    assert "Underexposed" in reason


def test_exposure_rejects_white_image():
    from winnow.quality import check_exposure
    white = np.full((100, 100, 3), 255, dtype=np.uint8)
    passed, reason = check_exposure(white)
    assert not passed
    assert "Overexposed" in reason


def test_exposure_passes_normal_image():
    from winnow.quality import check_exposure
    mid = np.full((100, 100, 3), 128, dtype=np.uint8)
    passed, _ = check_exposure(mid)
    assert passed


# ── check_face_size ───────────────────────────────────────────────────────────

def test_face_size_rejects_small_face():
    from winnow.quality import check_face_size
    passed, reason = check_face_size(30, 30, min_px=50)
    assert not passed
    assert "small" in reason


def test_face_size_passes_adequate_face():
    from winnow.quality import check_face_size
    passed, _ = check_face_size(100, 100, min_px=50)
    assert passed


def test_face_size_rejects_if_either_dimension_small():
    from winnow.quality import check_face_size
    passed, _ = check_face_size(100, 30, min_px=50)
    assert not passed


# ── check_confidence ──────────────────────────────────────────────────────────

def test_confidence_rejects_low_score():
    from winnow.quality import check_confidence
    passed, reason = check_confidence(0.5, min_conf=0.7)
    assert not passed
    assert "confidence" in reason.lower()


def test_confidence_passes_high_score():
    from winnow.quality import check_confidence
    passed, _ = check_confidence(0.95, min_conf=0.7)
    assert passed


def test_confidence_passes_none_score():
    from winnow.quality import check_confidence
    passed, _ = check_confidence(None, min_conf=0.7)
    assert passed


# ── assess_quality (integration) ─────────────────────────────────────────────

def test_assess_quality_passes_good_image():
    from winnow.quality import assess_quality
    img = _noisy_color_image()
    result = assess_quality(img, face_bbox=(10, 10, 110, 110), confidence=0.9)
    assert result.passed
    assert result.blur_score is not None
    assert result.blur_score > 0


def test_assess_quality_blur_score_is_low_for_flat_image():
    from winnow.quality import assess_quality
    flat = _rgb_image(128, 128, 128)
    result = assess_quality(flat)
    assert result.blur_score is not None
    assert result.blur_score < 1.0


def test_assess_quality_collects_multiple_failures():
    from winnow.quality import assess_quality
    black = _rgb_image(0, 0, 0)
    result = assess_quality(black, face_bbox=(0, 0, 10, 10), confidence=0.3)
    assert not result.passed
    assert len(result.reasons) >= 2


def test_assess_quality_skips_face_size_without_bbox():
    from winnow.quality import assess_quality
    img = _noisy_color_image()
    result = assess_quality(img, face_bbox=None, confidence=0.9)
    assert result.passed
