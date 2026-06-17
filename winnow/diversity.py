"""
Diversity selection for training data curation.

Selection pipeline:
1. Concurrent thumbnail download
2. Quality filtering (blur, IR, exposure, confidence, face size)
3. Face crop extraction (embed person's face, not full image)
4. Embedding computation (InsightFace)
5. Cluster-aware selection (K-Medoids + FPS with hard example weighting)
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import numpy as np
import requests
from PIL import Image

from .config import Config, get_headers
from .embeddings import get_embedding, is_embedding_available
from .quality import assess_quality

logger = logging.getLogger(__name__)

# Candidate pool: cap at _POOL_CAP assets, but take at least _POOL_SCALE × the
# requested limit so small limits don't artificially narrow the search space.
_POOL_CAP = 3000
_POOL_SCALE = 20

# Embedding batch size: bounds decoded thumbnails in memory.
# At ~3-8 MB each, 32 images ≈ 100–250 MB peak — safe in a 4 GB container.
_EMBEDDING_BATCH_SIZE = 32


def select_diverse_assets(
    assets: list,
    limit: int | str,
    entity_name: str,
    selection_mode: str = "smart",
    person_id: str | None = None,
    progress_callback=None,
    fetch_fn=None,
) -> list:
    """
    Select diverse assets using cluster-aware FPS or time spread.

    Args:
        assets: List of asset dicts from Immich API
        limit: Number to select, or "auto" for dynamic selection
        entity_name: Name of the person for logging
        selection_mode: 'smart' (embedding-based) or 'time' (time spread)
        progress_callback: Optional callback(current, total) for progress
        fetch_fn: Optional callable(asset_id) -> Image | None; defaults to
                  _fetch_thumbnail. Injected for testability.

    Returns:
        List of selected assets
    """
    # Fast path: fewer assets than limit — sort for consistent ordering with other paths
    if limit != "auto" and len(assets) <= limit:
        return sorted(assets, key=lambda x: x.get("fileCreatedAt", ""))

    # Sort by creation time
    assets = sorted(assets, key=lambda x: x.get("fileCreatedAt", ""))

    if selection_mode != "smart" or not is_embedding_available():
        if selection_mode == "smart":
            logger.warning("InsightFace unavailable. Falling back to time spread.")
        return _select_time_spread(assets, limit)

    try:
        return _select_by_embedding(assets, limit, person_id, progress_callback, fetch_fn=fetch_fn)
    except Exception as e:
        logger.error("Smart Diversity failed: %s. Falling back to time spread.", e)
        return _select_time_spread(assets, limit)


# =============================================================================
# Thumbnail & Metadata Helpers
# =============================================================================


def _fetch_thumbnail(asset_id: str, timeout: int = 10) -> Image.Image | None:
    """Fetch thumbnail from Immich API."""
    try:
        url = f"{Config.IMMICH_URL}/api/assets/{asset_id}/thumbnail?size=preview&format=JPEG"
        resp = requests.get(url, headers=get_headers(), timeout=timeout)
        return Image.open(BytesIO(resp.content)) if resp.ok else None
    except Exception:
        return None


def _get_face_bbox(asset: dict, person_id: str | None = None) -> tuple[float, float, float, float] | None:
    """Extract face bounding box from asset metadata for the given person."""
    for person in asset.get("people", []):
        if person_id and person.get("id") != person_id:
            continue
        faces = person.get("faces", [])
        if faces:
            f = faces[0]
            return (
                f.get("boundingBoxX1", 0),
                f.get("boundingBoxY1", 0),
                f.get("boundingBoxX2", 0),
                f.get("boundingBoxY2", 0),
            )
    return None


def _get_face_confidence(asset: dict, person_id: str | None = None) -> float | None:
    """Extract face detection confidence from asset metadata for the given person."""
    for person in asset.get("people", []):
        if person_id and person.get("id") != person_id:
            continue
        faces = person.get("faces", [])
        if faces:
            f = faces[0]
            score = f.get("score")
            return score if score is not None else f.get("confidence")
    return None


def _crop_face_from_thumbnail(
    img: Image.Image,
    asset: dict,
    margin: float = 0.25,
    person_id: str | None = None,
) -> Image.Image | None:
    """Crop the face region from a thumbnail using Immich bbox metadata.

    By cropping before embedding, we guarantee InsightFace embeds the
    correct person's face (not the largest face in a group photo).

    Args:
        img: Full preview thumbnail
        asset: Asset dict with people/faces metadata
        margin: Extra margin around the bbox (fraction, default 25%)
        person_id: If provided, only crop from this person's face data.

    Returns:
        Cropped face PIL image, or None if no face metadata available
    """
    bbox = _get_face_bbox(asset, person_id=person_id)
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    img_w, img_h = img.size

    # Get metadata dimensions to scale bbox — must match the same person as _get_face_bbox
    for person in asset.get("people", []):
        if person_id and person.get("id") != person_id:
            continue
        faces = person.get("faces", [])
        if faces:
            meta_w = faces[0].get("imageWidth") or img_w
            meta_h = faces[0].get("imageHeight") or img_h
            scale_x, scale_y = img_w / meta_w, img_h / meta_h
            x1, y1 = x1 * scale_x, y1 * scale_y
            x2, y2 = x2 * scale_x, y2 * scale_y
            break

    face_w, face_h = x2 - x1, y2 - y1

    # Add margin so InsightFace's internal alignment has context
    mx, my = face_w * margin, face_h * margin
    crop = img.crop(
        (
            max(0, x1 - mx),
            max(0, y1 - my),
            min(img_w, x2 + mx),
            min(img_h, y2 + my),
        )
    )

    # Skip if too small for meaningful embedding
    if crop.width < 30 or crop.height < 30:
        return None

    return crop


def _scale_bbox_to_thumbnail(
    bbox: tuple[float, float, float, float],
    img: Image.Image,
    asset: dict,
    person_id: str | None = None,
) -> tuple[float, float, float, float]:
    """Scale a face bbox from original detection-image space to thumbnail-pixel space."""
    x1, y1, x2, y2 = bbox
    img_w, img_h = img.size
    for person in asset.get("people", []):
        if person_id and person.get("id") != person_id:
            continue
        faces = person.get("faces", [])
        if faces:
            meta_w = faces[0].get("imageWidth") or 0
            meta_h = faces[0].get("imageHeight") or 0
            scale_x = img_w / meta_w if meta_w else 1.0
            scale_y = img_h / meta_h if meta_h else 1.0
            return (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
    return bbox


# =============================================================================
# Embedding Collection
# =============================================================================


def _select_by_embedding(
    assets: list,
    limit: int | str,
    person_id: str | None = None,
    progress_callback=None,
    fetch_fn=None,
) -> list:
    """Select assets using embedding-based cluster-aware FPS.

    Pipeline:
    1. Concurrent thumbnail download
    2. Quality filtering
    3. Face crop extraction
    4. Embedding computation
    5. Cluster-aware selection with hard example weighting
    """
    effective_limit = 30 if limit == "auto" else limit
    pool_size = min(_POOL_CAP, max(effective_limit * _POOL_SCALE, len(assets)))

    # Subsample if needed (evenly distributed in time)
    if len(assets) > pool_size:
        indices = np.linspace(0, len(assets) - 1, pool_size, dtype=int)
        candidates = [assets[i] for i in indices]
    else:
        candidates = assets

    # --- Phases 1-4: Batched download → quality filter → crop → embed ---
    # Process in bounded batches so at most _BATCH decoded images live in RAM
    # at once. With 472 candidates each thumbnail is ~3-8 MB decoded; loading
    # all at once easily exhausts a 4 GB container limit on CPU.
    # LIMITATION — thumbnail-resolution embeddings drive full-res crop selection:
    # diversity selection runs InsightFace on Immich preview thumbnails (~720p)
    # to avoid downloading full-res for every candidate, but the training crop
    # comes from the full-resolution original. Embeddings from thumbnails are
    # representative in practice, but heavy JPEG compression on a preview could
    # produce a subtly different embedding than the full-res version. For most
    # libraries this is negligible; it matters if Immich preview quality is low.
    _fetch = fetch_fn or _fetch_thumbnail
    embeddings, valid_candidates, confidence_scores = [], [], []
    quality_filtered = 0
    processed = 0

    for batch_start in range(0, len(candidates), _EMBEDDING_BATCH_SIZE):
        batch = candidates[batch_start : batch_start + _EMBEDDING_BATCH_SIZE]

        # Download this batch concurrently
        batch_images: dict[str, Image.Image] = {}
        with ThreadPoolExecutor(max_workers=min(8, len(batch))) as pool:
            futures = {pool.submit(_fetch, a["id"]): a for a in batch}
            for future in as_completed(futures):
                asset = futures[future]
                try:
                    img = future.result()
                    if img is not None:
                        batch_images[asset["id"]] = img
                except Exception as e:
                    logger.debug("Failed to fetch thumbnail for %s: %s", asset["id"], e)
                    continue

        # Process each image; batch_images goes out of scope after this loop,
        # bounding peak thumbnail memory to _BATCH images per iteration.
        for asset in batch:
            img = batch_images.get(asset["id"])
            processed += 1
            if progress_callback:
                progress_callback(processed, len(candidates))
            if img is None:
                continue

            confidence = _get_face_confidence(asset, person_id=person_id)

            face_bbox = _get_face_bbox(asset, person_id=person_id)
            thumbnail_bbox = (
                _scale_bbox_to_thumbnail(face_bbox, img, asset, person_id)
                if face_bbox is not None else None
            )
            quality = assess_quality(
                img,
                face_bbox=thumbnail_bbox,
                confidence=confidence,
                blur_threshold=Config.BLUR_THRESHOLD,
                min_face_px=Config.MIN_FACE_WIDTH,
                min_confidence=Config.MIN_CONFIDENCE,
            )
            if not quality.passed:
                quality_filtered += 1
                logger.debug("Quality filtered %s: %s", asset["id"], quality.reason)
                continue

            asset["quality_score"] = quality.blur_score
            face_crop = _crop_face_from_thumbnail(img, asset, person_id=person_id)
            embed_img = face_crop if face_crop is not None else img

            emb = get_embedding(embed_img, asset_id=asset["id"])
            if emb is not None:
                if np.linalg.norm(emb) < 1e-6:
                    logger.debug("Zero-norm embedding for asset %s, skipping", asset["id"])
                    continue
                embeddings.append(emb)
                valid_candidates.append(asset)
                confidence_scores.append(confidence)

    if quality_filtered > 0:
        logger.info("Quality filtering removed %s images.", quality_filtered)

    if not embeddings:
        logger.warning("No valid embeddings found. Falling back to time spread.")
        return _select_time_spread(assets, limit)

    if limit != "auto" and len(valid_candidates) < limit:
        logger.warning("Only %s valid embeddings. Returning all.", len(valid_candidates))
        return valid_candidates

    # --- Phase 5: Near-duplicate removal ---
    # Burst shots and repeated near-identical photos produce embeddings that are
    # close but not identical, so FPS doesn't filter them out on its own.
    # Greedily drop any candidate within DEDUP_THRESHOLD cosine distance of a
    # higher-quality image already in the kept set.
    embeddings, valid_candidates, confidence_scores = _dedup_embeddings(
        embeddings, valid_candidates, confidence_scores
    )

    # Re-check after dedup: pool may have shrunk below limit
    if limit != "auto" and len(valid_candidates) < limit:
        logger.warning("Only %s embeddings after near-duplicate removal. Returning all.", len(valid_candidates))
        return valid_candidates

    # --- Phase 6: Cluster-aware selection ---
    return _cluster_aware_selection(
        embeddings,
        valid_candidates,
        limit,
        confidence_scores=confidence_scores,
    )


# =============================================================================
# Near-Duplicate Removal
# =============================================================================

_DEDUP_THRESHOLD = 0.20  # cosine distance — burst shots ~0.01-0.05, same-event similar shots ~0.10-0.20


def _dedup_embeddings(
    embeddings: list,
    candidates: list,
    confidence_scores: list,
) -> tuple[list, list, list]:
    """Greedy near-duplicate removal before clustering.

    Sorts by quality score descending (best first), then for each candidate
    drops it if any already-kept embedding is within _DEDUP_THRESHOLD cosine
    distance. This eliminates burst-shot near-duplicates while preserving the
    highest-quality representative from each near-identical group.
    """
    if len(embeddings) < 2:
        return embeddings, candidates, confidence_scores

    emb_matrix = np.vstack(embeddings)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    emb_normed = emb_matrix / np.maximum(norms, 1e-8)

    # Sort by quality descending so the best image in each near-duplicate group wins.
    # Use explicit None check so a legitimate quality_score=0.0 isn't treated as missing.
    quality_scores = [qs if (qs := c.get("quality_score")) is not None else 0.0 for c in candidates]
    order = sorted(range(len(candidates)), key=lambda i: quality_scores[i], reverse=True)

    kept_indices = []
    # Pre-allocate a max-size buffer and fill row-by-row — eliminates the O(K²)
    # copy overhead from vstack-on-keep while keeping identical arithmetic.
    kept_buf = np.empty((len(order), emb_normed.shape[1]), dtype=emb_normed.dtype)
    n_kept = 0

    for i in order:
        if n_kept > 0:
            sims = emb_normed[i] @ kept_buf[:n_kept].T
            if np.any(sims > 1 - _DEDUP_THRESHOLD):
                continue
        kept_buf[n_kept] = emb_normed[i]
        n_kept += 1
        kept_indices.append(i)

    dropped = len(embeddings) - len(kept_indices)
    if dropped:
        logger.info("Near-duplicate removal dropped %s images (threshold %s).", dropped, _DEDUP_THRESHOLD)

    return (
        [embeddings[i] for i in kept_indices],
        [candidates[i] for i in kept_indices],
        [confidence_scores[i] for i in kept_indices],
    )


# =============================================================================
# K-Medoids (Lightweight Implementation)
# =============================================================================


def _kmedoids(dist_matrix: np.ndarray, k: int, max_iter: int = 50) -> tuple[list[int], np.ndarray]:
    """Lightweight K-Medoids clustering using cosine distance matrix.

    Args:
        dist_matrix: (N, N) pairwise distance matrix
        k: Number of clusters
        max_iter: Maximum iterations for swap step

    Returns:
        (medoid_indices, cluster_labels) tuple
    """
    n = dist_matrix.shape[0]
    rng = np.random.default_rng(42)

    # Initialize medoids: first = most central point, rest = farthest from chosen
    total_dist = dist_matrix.sum(axis=1)
    medoids = [int(np.argmin(total_dist))]

    for _ in range(k - 1):
        dists_to_chosen = dist_matrix[:, medoids].min(axis=1)
        dists_to_chosen[medoids] = -np.inf
        medoids.append(int(np.argmax(dists_to_chosen)))

    # Iterative swap step
    medoids = list(medoids)
    labels = np.argmin(dist_matrix[:, medoids], axis=1)
    cost = dist_matrix[np.arange(n), np.array(medoids)[labels]].sum()

    for _ in range(max_iter):
        improved = False
        # Try swapping each medoid with a random non-medoid
        medoid_set = set(medoids)
        non_medoids = [i for i in range(n) if i not in medoid_set]
        if not non_medoids:
            break

        for m_idx in range(k):
            candidates = rng.choice(non_medoids, size=min(10, len(non_medoids)), replace=False)
            for cand in candidates:
                new_medoids = medoids.copy()
                new_medoids[m_idx] = cand
                new_labels = np.argmin(dist_matrix[:, new_medoids], axis=1)
                new_cost = dist_matrix[np.arange(n), np.array(new_medoids)[new_labels]].sum()
                if new_cost < cost:
                    medoids = new_medoids
                    labels = new_labels
                    cost = new_cost
                    improved = True
                    break
            if improved:
                break

        if not improved:
            break

    return medoids, labels


# =============================================================================
# Cluster-Aware Selection (K-Medoids + FPS Hybrid)
# =============================================================================


def _compute_adaptive_threshold(emb_normed: np.ndarray) -> float:
    """Compute adaptive FPS stop threshold based on actual embedding distribution.

    Instead of a hardcoded threshold, samples pairwise distances and sets
    the threshold as 20% of the median pairwise distance.
    """
    n = len(emb_normed)
    sample_size = min(200, n)
    rng = np.random.default_rng(42)
    indices = rng.choice(n, sample_size, replace=False) if n > sample_size else np.arange(n)
    sample = emb_normed[indices]

    pairwise = 1 - sample @ sample.T
    upper_tri = pairwise[np.triu_indices(len(sample), k=1)]
    if len(upper_tri) == 0:
        return 0.05
    median_dist = float(np.median(upper_tri))
    threshold = max(0.05, median_dist * 0.20)

    logger.debug("Adaptive threshold: %.4f (median_dist=%.4f)", threshold, median_dist)
    return threshold


def _cluster_aware_selection(
    embeddings: list,
    candidates: list,
    limit: int | str,
    confidence_scores: list | None = None,
) -> list:
    """Two-stage selection: K-Medoids clustering → FPS with hard example weighting.

    Stage 1: Cluster embeddings into k groups, select medoids as initial picks.
             This guarantees at least one representative from every distinct "look".

    Stage 2: Fill remaining budget with FPS across cluster boundaries,
             biasing toward hard examples (low-confidence candidates).
    """
    emb_matrix = np.vstack(embeddings)  # (N, D)
    n = len(emb_matrix)

    # Normalize for cosine distance
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    emb_normed = emb_matrix / np.maximum(norms, 1e-8)

    # Build confidence weight array for hard example boosting.
    # Default to 1.0 for faces with no confidence score: treat as high-confidence
    # (no boost) rather than hard-example territory. A missing score field should
    # not cause these images to beat genuinely high-confidence detections in FPS.
    conf_array = np.ones(n)
    if confidence_scores:
        for i, c in enumerate(confidence_scores):
            if c is not None:
                conf_array[i] = c

    # Compute adaptive threshold for auto mode
    auto_threshold = _compute_adaptive_threshold(emb_normed) if limit == "auto" else 0.0
    target = Config.MAX_AUTO_IMAGES if limit == "auto" else limit

    # Short-circuit: nothing to select
    if limit != "auto" and target <= 0:
        return []

    # --- Stage 1: K-Medoids clustering ---
    # Cap k at target so we never seed more cluster representatives than requested.
    k = min(max(5, target // 4), max(1, n // 3), n, target)  # e.g., 1-20 clusters
    logger.debug("Clustering %s embeddings into %s groups (K-Medoids)...", n, k)

    # Compute full cosine distance matrix
    dist_matrix = 1 - emb_normed @ emb_normed.T

    medoid_indices, cluster_labels = _kmedoids(dist_matrix, k)
    selected = list(medoid_indices)

    logger.debug("Selected %s cluster medoids as initial picks.", len(selected))

    # --- Stage 2: FPS with hard example weighting ---
    min_dists = np.full(n, np.inf)

    # Initialize min distances from all medoids
    for idx in selected:
        dists = dist_matrix[idx]
        min_dists = np.minimum(min_dists, dists)
    for idx in selected:
        min_dists[idx] = -np.inf

    # Hard example weighting: boost distance for low-confidence candidates.
    # conf_array is constant after this point, so compute once outside the loop.
    hard_weight = np.where(conf_array < 0.85, 1.0 + (0.85 - conf_array) * 2.0, 1.0)

    while len(selected) < target:
        weighted_dists = min_dists * hard_weight

        best_idx = int(np.argmax(weighted_dists))
        best_dist = min_dists[best_idx]  # Use unweighted for threshold comparison

        if best_dist == -np.inf:
            break  # All points selected

        if limit == "auto" and best_dist < auto_threshold:
            logger.debug(
                f"Auto-stop: next best image {best_dist:.3f} away (adaptive threshold {auto_threshold:.4f})."
            )
            break

        selected.append(best_idx)

        # Update min distances
        dists_to_new = dist_matrix[best_idx]
        min_dists = np.minimum(min_dists, dists_to_new)
        min_dists[best_idx] = -np.inf

    hard_count = sum(
        1 for i in selected
        if confidence_scores
        and i < len(confidence_scores)
        and confidence_scores[i] is not None
        and confidence_scores[i] < 0.85
    )
    logger.info("Selection complete: %s images (%s hard examples with confidence < 0.85).", len(selected), hard_count)

    # Slice to target: the while loop enforces this for non-auto mode, but
    # guard here too in case the medoid seed already exceeded target (small target).
    result = [candidates[i] for i in selected]
    if limit != "auto":
        result = result[:target]
    return result


# =============================================================================
# Time Spread Fallback
# =============================================================================


def _select_time_spread(assets: list, limit: int | str) -> list:
    """Select N assets evenly distributed in time."""
    if limit == "auto":
        limit = 30

    logger.info("Selecting %s images using time spread.", limit)

    if len(assets) <= limit:
        return assets

    indices = np.linspace(0, len(assets) - 1, limit, dtype=int)
    return [assets[i] for i in indices]
