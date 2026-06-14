"""Tests for core diversity selection algorithms (pure functions, no network)."""

import numpy as np
import pytest
from PIL import Image


def _unit_embeddings(n: int, d: int = 512, seed: int = 42) -> list:
    """Return n normalised random embeddings — well-separated in high-dim space."""
    rng = np.random.default_rng(seed)
    embs = rng.standard_normal((n, d)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    return list(embs)


def _asset_with_face(
    asset_id="a1", person_id="p1",
    x1=10, y1=10, x2=60, y2=60,
    img_w=100, img_h=100, score=0.9,
):
    return {
        "id": asset_id,
        "people": [{
            "id": person_id,
            "faces": [{
                "boundingBoxX1": x1, "boundingBoxY1": y1,
                "boundingBoxX2": x2, "boundingBoxY2": y2,
                "imageWidth": img_w, "imageHeight": img_h,
                "score": score,
            }],
        }],
    }


# ── _get_face_bbox ─────────────────────────────────────────────────────────────

def test_get_face_bbox_returns_coords():
    from winnow.diversity import _get_face_bbox
    asset = _asset_with_face(x1=5, y1=10, x2=55, y2=70)
    assert _get_face_bbox(asset) == (5, 10, 55, 70)


def test_get_face_bbox_filters_by_person_id():
    from winnow.diversity import _get_face_bbox
    asset = _asset_with_face(person_id="p1")
    assert _get_face_bbox(asset, person_id="p999") is None


def test_get_face_bbox_returns_none_empty_faces():
    from winnow.diversity import _get_face_bbox
    asset = {"id": "a1", "people": [{"id": "p1", "faces": []}]}
    assert _get_face_bbox(asset) is None


def test_get_face_bbox_returns_none_no_people():
    from winnow.diversity import _get_face_bbox
    assert _get_face_bbox({"id": "a1"}) is None


# ── _get_face_confidence ───────────────────────────────────────────────────────

def test_get_face_confidence_returns_score():
    from winnow.diversity import _get_face_confidence
    asset = _asset_with_face(score=0.92)
    assert _get_face_confidence(asset) == pytest.approx(0.92)


def test_get_face_confidence_filters_by_person_id():
    from winnow.diversity import _get_face_confidence
    asset = _asset_with_face(person_id="p1", score=0.9)
    assert _get_face_confidence(asset, person_id="p999") is None


def test_get_face_confidence_returns_none_empty_faces():
    from winnow.diversity import _get_face_confidence
    asset = {"id": "a1", "people": [{"id": "p1", "faces": []}]}
    assert _get_face_confidence(asset) is None


# ── _crop_face_from_thumbnail ──────────────────────────────────────────────────

def test_crop_face_returns_image():
    from winnow.diversity import _crop_face_from_thumbnail
    img = Image.new("RGB", (200, 200), color=(128, 64, 32))
    asset = _asset_with_face(x1=50, y1=50, x2=150, y2=150, img_w=200, img_h=200)
    crop = _crop_face_from_thumbnail(img, asset)
    assert crop is not None
    assert crop.width > 0 and crop.height > 0


def test_crop_face_scales_bbox_to_thumbnail():
    """When thumbnail is half the metadata dimensions, bbox is scaled accordingly."""
    from winnow.diversity import _crop_face_from_thumbnail
    img = Image.new("RGB", (200, 200))
    # Metadata says 400×400; bbox covers the centre quarter
    asset = _asset_with_face(x1=100, y1=100, x2=300, y2=300, img_w=400, img_h=400)
    crop = _crop_face_from_thumbnail(img, asset)
    assert crop is not None
    assert crop.width <= 200 and crop.height <= 200


def test_crop_face_returns_none_no_metadata():
    from winnow.diversity import _crop_face_from_thumbnail
    img = Image.new("RGB", (100, 100))
    assert _crop_face_from_thumbnail(img, {"id": "a1"}) is None


def test_crop_face_returns_none_for_sub_30px_bbox():
    """A 1×1 bbox produces a crop too small to embed — should be rejected."""
    from winnow.diversity import _crop_face_from_thumbnail
    img = Image.new("RGB", (100, 100))
    asset = _asset_with_face(x1=50, y1=50, x2=51, y2=51, img_w=100, img_h=100)
    assert _crop_face_from_thumbnail(img, asset) is None


def test_crop_face_respects_person_id_filter():
    from winnow.diversity import _crop_face_from_thumbnail
    img = Image.new("RGB", (200, 200))
    asset = _asset_with_face(person_id="p1", x1=50, y1=50, x2=150, y2=150)
    assert _crop_face_from_thumbnail(img, asset, person_id="p999") is None


# ── _dedup_embeddings ──────────────────────────────────────────────────────────

def test_dedup_keeps_all_diverse_embeddings():
    from winnow.diversity import _dedup_embeddings
    embs = _unit_embeddings(20)
    candidates = [{"id": str(i)} for i in range(20)]
    _, out_cands, _ = _dedup_embeddings(embs, candidates, [None] * 20)
    # Random 512-dim unit vectors are far apart — all should survive
    assert len(out_cands) == 20


def test_dedup_removes_near_duplicate():
    from winnow.diversity import _dedup_embeddings
    base = np.zeros(512, dtype=np.float32)
    base[0] = 1.0
    # Cosine distance ≈ 0.01 — well within the 0.20 dedup threshold
    near_dup = base.copy()
    near_dup[1] = 0.014
    near_dup /= np.linalg.norm(near_dup)

    embs = [base, near_dup]
    candidates = [{"id": "base", "quality_score": 0.9}, {"id": "dup", "quality_score": 0.5}]
    _, out_cands, _ = _dedup_embeddings(embs, candidates, [None, None])
    assert len(out_cands) == 1
    assert out_cands[0]["id"] == "base"


def test_dedup_keeps_higher_quality_from_duplicate_pair():
    from winnow.diversity import _dedup_embeddings
    base = np.zeros(512, dtype=np.float32)
    base[0] = 1.0
    near_dup = base.copy()
    near_dup[1] = 0.014
    near_dup /= np.linalg.norm(near_dup)

    # Reversed quality: near_dup is sharper
    embs = [base, near_dup]
    candidates = [{"id": "base", "quality_score": 0.3}, {"id": "dup", "quality_score": 0.95}]
    _, out_cands, _ = _dedup_embeddings(embs, candidates, [None, None])
    assert len(out_cands) == 1
    assert out_cands[0]["id"] == "dup"


def test_dedup_single_embedding_passes_through():
    from winnow.diversity import _dedup_embeddings
    embs = _unit_embeddings(1)
    out_embs, out_cands, _ = _dedup_embeddings(embs, [{"id": "only"}], [None])
    assert len(out_cands) == 1


def test_dedup_treats_zero_quality_score_as_zero_not_missing():
    """quality_score=0.0 is a valid score — should not be treated as absent."""
    from winnow.diversity import _dedup_embeddings
    base = np.zeros(512, dtype=np.float32)
    base[0] = 1.0
    near_dup = base.copy()
    near_dup[1] = 0.014
    near_dup /= np.linalg.norm(near_dup)

    embs = [base, near_dup]
    # base has explicit 0.0; near_dup has 0.5 — near_dup should win
    candidates = [{"id": "base", "quality_score": 0.0}, {"id": "dup", "quality_score": 0.5}]
    _, out_cands, _ = _dedup_embeddings(embs, candidates, [None, None])
    assert out_cands[0]["id"] == "dup"


# ── _kmedoids ──────────────────────────────────────────────────────────────────

def _dist_matrix(embs):
    m = np.vstack(embs)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    return 1 - m @ m.T


def test_kmedoids_returns_k_distinct_medoids():
    from winnow.diversity import _kmedoids
    dist = _dist_matrix(_unit_embeddings(30))
    medoids, _ = _kmedoids(dist, k=5)
    assert len(medoids) == 5
    assert len(set(medoids)) == 5


def test_kmedoids_labels_cover_all_points():
    from winnow.diversity import _kmedoids
    dist = _dist_matrix(_unit_embeddings(20))
    medoids, labels = _kmedoids(dist, k=4)
    assert len(labels) == 20
    assert set(labels).issubset(set(range(4)))


def test_kmedoids_medoids_are_valid_indices():
    from winnow.diversity import _kmedoids
    n = 15
    dist = _dist_matrix(_unit_embeddings(n))
    medoids, _ = _kmedoids(dist, k=3)
    assert all(0 <= m < n for m in medoids)


def test_kmedoids_k_equals_n_selects_all():
    from winnow.diversity import _kmedoids
    n = 5
    dist = _dist_matrix(_unit_embeddings(n))
    medoids, _ = _kmedoids(dist, k=n)
    assert len(medoids) == n


# ── _compute_adaptive_threshold ────────────────────────────────────────────────

def test_adaptive_threshold_positive():
    from winnow.diversity import _compute_adaptive_threshold
    embs = np.array(_unit_embeddings(50))
    assert _compute_adaptive_threshold(embs) > 0


def test_adaptive_threshold_floor_for_identical_embeddings():
    """All-identical embeddings → median pairwise distance = 0 → floor at 0.05."""
    from winnow.diversity import _compute_adaptive_threshold
    base = np.zeros((10, 512), dtype=np.float32)
    base[:, 0] = 1.0
    assert _compute_adaptive_threshold(base) == pytest.approx(0.05)


def test_adaptive_threshold_single_point_returns_floor():
    from winnow.diversity import _compute_adaptive_threshold
    single = np.ones((1, 512), dtype=np.float32)
    single /= np.linalg.norm(single)
    assert _compute_adaptive_threshold(single) == pytest.approx(0.05)


def test_adaptive_threshold_scales_with_spread():
    """A more spread-out embedding set should produce a higher threshold."""
    from winnow.diversity import _compute_adaptive_threshold
    tight = np.array(_unit_embeddings(30, seed=0)) * 0.001 + np.array([1.0] + [0.0] * 511)
    tight /= np.linalg.norm(tight, axis=1, keepdims=True)
    diverse = np.array(_unit_embeddings(30, seed=1))
    assert _compute_adaptive_threshold(diverse) > _compute_adaptive_threshold(tight)


# ── _select_time_spread ────────────────────────────────────────────────────────

def test_time_spread_returns_exact_n():
    from winnow.diversity import _select_time_spread
    assets = [{"id": str(i)} for i in range(100)]
    assert len(_select_time_spread(assets, limit=10)) == 10


def test_time_spread_returns_all_when_under_limit():
    from winnow.diversity import _select_time_spread
    assets = [{"id": str(i)} for i in range(5)]
    assert len(_select_time_spread(assets, limit=20)) == 5


def test_time_spread_auto_defaults_to_30():
    from winnow.diversity import _select_time_spread
    assets = [{"id": str(i)} for i in range(200)]
    assert len(_select_time_spread(assets, limit="auto")) == 30


def test_time_spread_includes_first_and_last():
    from winnow.diversity import _select_time_spread
    assets = [{"id": str(i)} for i in range(100)]
    result = _select_time_spread(assets, limit=5)
    ids = [int(a["id"]) for a in result]
    assert ids[0] == 0
    assert ids[-1] == 99


# ── _cluster_aware_selection ───────────────────────────────────────────────────

def test_cluster_selection_returns_exact_limit(monkeypatch):
    from winnow.diversity import _cluster_aware_selection
    monkeypatch.setattr("winnow.diversity.Config.MAX_AUTO_IMAGES", 20)
    embs = _unit_embeddings(50)
    candidates = [{"id": str(i)} for i in range(50)]
    result = _cluster_aware_selection(embs, candidates, limit=10)
    assert len(result) == 10


def test_cluster_selection_output_is_subset_of_input(monkeypatch):
    from winnow.diversity import _cluster_aware_selection
    monkeypatch.setattr("winnow.diversity.Config.MAX_AUTO_IMAGES", 20)
    embs = _unit_embeddings(30)
    candidates = [{"id": str(i)} for i in range(30)]
    result = _cluster_aware_selection(embs, candidates, limit=10)
    result_ids = {a["id"] for a in result}
    assert result_ids.issubset({a["id"] for a in candidates})


def test_cluster_selection_auto_stops_early_on_tight_cluster(monkeypatch):
    """When all embeddings are nearly identical auto mode should stop early."""
    from winnow.diversity import _cluster_aware_selection
    monkeypatch.setattr("winnow.diversity.Config.MAX_AUTO_IMAGES", 20)
    rng = np.random.default_rng(0)
    base = np.zeros(512, dtype=np.float32)
    base[0] = 1.0
    embs = []
    for _ in range(50):
        v = base + rng.standard_normal(512).astype(np.float32) * 0.001
        v /= np.linalg.norm(v)
        embs.append(v)
    candidates = [{"id": str(i)} for i in range(50)]
    result = _cluster_aware_selection(list(embs), candidates, limit="auto")
    assert len(result) < 20


def test_cluster_selection_hard_example_weighting_accepted(monkeypatch):
    """Confidence scores are accepted without error."""
    from winnow.diversity import _cluster_aware_selection
    monkeypatch.setattr("winnow.diversity.Config.MAX_AUTO_IMAGES", 20)
    embs = _unit_embeddings(20)
    candidates = [{"id": str(i)} for i in range(20)]
    conf = [0.7 if i % 2 == 0 else 0.95 for i in range(20)]
    result = _cluster_aware_selection(embs, candidates, limit=5, confidence_scores=conf)
    assert len(result) == 5
