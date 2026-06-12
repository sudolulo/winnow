"""
Diversity selection for training data curation.

Selection pipeline:
1. Concurrent thumbnail download
2. Quality filtering (blur, IR, exposure, confidence, face size)
3. Face crop extraction (embed person's face, not full image)
4. Embedding computation (InsightFace or SigLIP)
5. Cluster-aware selection (K-Medoids + FPS with hard example weighting)
"""

import logging
from io import BytesIO

import numpy as np
import requests
from PIL import Image

from .config import Config, get_headers
from .embeddings import get_embedding, is_embedding_available
from .quality import assess_quality

logger = logging.getLogger(__name__)


def select_diverse_assets(
    assets: list,
    limit: int | str,
    entity_name: str,
    selection_mode: str = "smart",
    entity_type: str = "face",
    person_id: str | None = None,
    progress_callback=None,
) -> list:
    """
    Select diverse assets using cluster-aware FPS or time spread.

    Args:
        assets: List of asset dicts from Immich API
        limit: Number to select, or "auto" for dynamic selection
        entity_name: Name of the person/object for logging
        selection_mode: 'smart' (embedding-based) or 'time' (time spread)
        entity_type: 'face' or 'object' - determines embedding model
        progress_callback: Optional callback(current, total) for progress

    Returns:
        List of selected assets
    """
    # Fast path: fewer assets than limit
    if limit != "auto" and len(assets) <= limit:
        return assets

    # Sort by creation time
    assets = sorted(assets, key=lambda x: x.get("fileCreatedAt", ""))

    if selection_mode != "smart" or not is_embedding_available(entity_type):
        if selection_mode == "smart":
            model_name = "InsightFace" if entity_type == "face" else "SigLIP"
            logger.warning(f"{model_name} unavailable. Falling back to time spread.")
        return _select_time_spread(assets, limit)

    try:
        return _select_by_embedding(assets, limit, entity_type, person_id, progress_callback)
    except Exception as e:
        logger.error(f"Smart Diversity failed: {e}. Falling back to time spread.")
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


# =============================================================================
# Embedding Collection
# =============================================================================


def _select_by_embedding(
    assets: list,
    limit: int | str,
    entity_type: str,
    person_id: str | None = None,
    progress_callback=None,
) -> list:
    """Select assets using embedding-based cluster-aware FPS.

    Pipeline:
    1. Concurrent thumbnail download
    2. Quality filtering
    3. Face crop extraction (face mode only)
    4. Embedding computation
    5. Cluster-aware selection with hard example weighting
    """
    # Determine candidate pool (cap at 3000 for performance)
    effective_limit = 30 if limit == "auto" else limit
    pool_size = min(3000, max(effective_limit * 20, len(assets)))

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
    from concurrent.futures import ThreadPoolExecutor, as_completed

    _BATCH = 32
    embeddings, valid_candidates, confidence_scores = [], [], []
    quality_filtered = 0
    processed = 0

    for batch_start in range(0, len(candidates), _BATCH):
        batch = candidates[batch_start : batch_start + _BATCH]

        # Download this batch concurrently
        batch_images: dict[str, Image.Image] = {}
        with ThreadPoolExecutor(max_workers=min(8, len(batch))) as pool:
            futures = {pool.submit(_fetch_thumbnail, a["id"]): a for a in batch}
            for future in as_completed(futures):
                asset = futures[future]
                try:
                    img = future.result()
                    if img is not None:
                        batch_images[asset["id"]] = img
                except Exception as e:
                    logger.debug(f"Failed to fetch thumbnail for {asset['id']}: {e}")
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

            if entity_type == "face":
                face_bbox = _get_face_bbox(asset, person_id=person_id)
                quality = assess_quality(
                    img,
                    face_bbox=face_bbox,
                    confidence=confidence,
                    blur_threshold=Config.BLUR_THRESHOLD,
                    min_face_px=Config.MIN_FACE_WIDTH,
                    min_confidence=Config.MIN_CONFIDENCE,
                )
                if not quality.passed:
                    quality_filtered += 1
                    logger.debug(f"Quality filtered {asset['id']}: {quality.reason}")
                    continue

                face_crop = _crop_face_from_thumbnail(img, asset, person_id=person_id)
                embed_img = face_crop if face_crop is not None else img
            else:
                embed_img = img

            emb = get_embedding(embed_img, entity_type, asset_id=asset["id"])
            if emb is not None:
                embeddings.append(emb)
                valid_candidates.append(asset)
                confidence_scores.append(confidence)

    if quality_filtered > 0:
        logger.info(f"Quality filtering removed {quality_filtered} images.")

    if not embeddings:
        logger.warning("No valid embeddings found. Falling back to time spread.")
        return _select_time_spread(assets, limit)

    if limit != "auto" and len(valid_candidates) < limit:
        logger.warning(f"Only {len(valid_candidates)} valid embeddings. Returning all.")
        return valid_candidates

    # --- Phase 5: Cluster-aware selection ---
    return _cluster_aware_selection(
        embeddings,
        valid_candidates,
        limit,
        entity_type=entity_type,
        confidence_scores=confidence_scores,
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
    cost = sum(dist_matrix[i, medoids[labels[i]]] for i in range(n))

    for _ in range(max_iter):
        improved = False
        # Try swapping each medoid with a random non-medoid
        non_medoids = [i for i in range(n) if i not in medoids]
        if not non_medoids:
            break

        for m_idx in range(k):
            candidates = rng.choice(non_medoids, size=min(10, len(non_medoids)), replace=False)
            for cand in candidates:
                new_medoids = medoids.copy()
                new_medoids[m_idx] = cand
                new_labels = np.argmin(dist_matrix[:, new_medoids], axis=1)
                new_cost = sum(dist_matrix[i, new_medoids[new_labels[i]]] for i in range(n))
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


def _compute_adaptive_threshold(emb_normed: np.ndarray, entity_type: str) -> float:
    """Compute adaptive FPS stop threshold based on actual embedding distribution.

    Instead of a hardcoded threshold, samples pairwise distances and sets
    the threshold as a fraction of the median pairwise distance.
    """
    n = len(emb_normed)
    sample_size = min(200, n)
    rng = np.random.default_rng(42)
    indices = rng.choice(n, sample_size, replace=False) if n > sample_size else np.arange(n)
    sample = emb_normed[indices]

    # Compute pairwise cosine distances for the sample
    pairwise = 1 - sample @ sample.T
    upper_tri = pairwise[np.triu_indices(len(sample), k=1)]
    median_dist = float(np.median(upper_tri))

    # Faces: 20% of median (tighter — want fewer, more distinct images)
    # Objects: 10% of median (wider — want more diversity)
    fraction = 0.20 if entity_type == "face" else 0.10
    threshold = max(0.05, median_dist * fraction)

    logger.debug(
        f"Adaptive threshold: {threshold:.4f} "
        f"(median_dist={median_dist:.4f}, fraction={fraction}, type={entity_type})"
    )
    return threshold


def _cluster_aware_selection(
    embeddings: list,
    candidates: list,
    limit: int | str,
    entity_type: str = "face",
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

    # Build confidence weight array for hard example boosting
    conf_array = np.ones(n)
    if confidence_scores and entity_type == "face":
        for i, c in enumerate(confidence_scores):
            if c is not None:
                conf_array[i] = c

    # Compute adaptive threshold for auto mode
    auto_threshold = _compute_adaptive_threshold(emb_normed, entity_type) if limit == "auto" else 0.0
    target = Config.MAX_AUTO_IMAGES if limit == "auto" else limit

    # --- Stage 1: K-Medoids clustering ---
    k = min(max(5, target // 4), n // 3, n)  # e.g., 5-20 clusters
    logger.debug(f"Clustering {n} embeddings into {k} groups (K-Medoids)...")

    # Compute full cosine distance matrix
    dist_matrix = 1 - emb_normed @ emb_normed.T

    medoid_indices, cluster_labels = _kmedoids(dist_matrix, k)
    selected = list(medoid_indices)
    selected_set = set(selected)

    logger.debug(f"Selected {len(selected)} cluster medoids as initial picks.")

    # --- Stage 2: FPS with hard example weighting ---
    min_dists = np.full(n, np.inf)

    # Initialize min distances from all medoids
    for idx in selected:
        dists = dist_matrix[idx]
        min_dists = np.minimum(min_dists, dists)
    for idx in selected:
        min_dists[idx] = -np.inf

    while len(selected) < target:
        # Hard example weighting: boost distance for low-confidence candidates
        # Confidence < 0.85 gets up to 1.5× distance boost
        hard_weight = np.where(conf_array < 0.85, 1.0 + (0.85 - conf_array) * 2.0, 1.0)
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
        selected_set.add(best_idx)

        # Update min distances
        dists_to_new = dist_matrix[best_idx]
        min_dists = np.minimum(min_dists, dists_to_new)
        min_dists[best_idx] = -np.inf

    # Log hard example stats
    if entity_type == "face":
        selected_conf = [conf_array[i] for i in selected if conf_array[i] < 1.0]
        hard_count = sum(1 for c in selected_conf if c < 0.85)
        logger.info(
            f"Selection complete: {len(selected)} images " f"({hard_count} hard examples with confidence < 0.85)."
        )
    else:
        logger.info(f"Selection complete: {len(selected)} diverse images.")

    return [candidates[i] for i in selected]


# =============================================================================
# Time Spread Fallback
# =============================================================================


def _select_time_spread(assets: list, limit: int | str) -> list:
    """Select N assets evenly distributed in time."""
    if limit == "auto":
        limit = 30

    logger.info(f"Selecting {limit} images using time spread.")

    if len(assets) <= limit:
        return assets

    indices = np.linspace(0, len(assets) - 1, limit, dtype=int)
    return [assets[i] for i in np.unique(indices)]
