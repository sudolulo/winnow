"""Immich API client for fetching people, assets, and face data."""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO

import requests
from PIL import Image, ImageOps

from .config import Config, get_headers

logger = logging.getLogger(__name__)

MAX_PAGES = 1000  # Safety limit for pagination
_MAX_ASSETS_PER_PERSON = 5000  # Stop fetching after this many — diversity pool is capped at 3000 anyway


@dataclass
class FaceData:
    """Pre-computed face data from Immich."""

    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float | None
    image_width: int
    image_height: int


def get_immich_version() -> tuple[int, int, int] | None:
    """Fetch Immich server version from GET /api/server/version.

    Returns (major, minor, patch) or None if unreachable or unparseable.
    """
    try:
        resp = requests.get(
            f"{Config.IMMICH_URL}/api/server/version",
            headers=get_headers(),
            timeout=5,
        )
        if resp.ok:
            data = resp.json()
            return (int(data["major"]), int(data["minor"]), int(data["patch"]))
        return None
    except Exception:
        return None


def get_people() -> list[dict]:
    """Fetch all people from Immich."""
    try:
        resp = requests.get(
            f"{Config.IMMICH_URL}/api/people",
            headers=get_headers(),
            timeout=10,
        )
        if resp.status_code == 401:
            logger.error("Immich API key is invalid or expired (401 Unauthorized). Update API_KEY.")
            return []
        resp.raise_for_status()
        return resp.json().get("people", [])
    except (requests.RequestException, ValueError) as e:
        logger.error("Failed to fetch people from Immich: %s", e)
        return []


def merge_people(survivor_id: str, merge_ids: list[str]) -> bool:
    """Merge duplicate people into survivor via Immich's merge endpoint.

    The survivor (identified by survivor_id) absorbs all faces and assets
    from the people in merge_ids, which are then removed from Immich.
    """
    try:
        resp = requests.put(
            f"{Config.IMMICH_URL}/api/people/{survivor_id}/merge",
            headers={**get_headers(), "Content-Type": "application/json"},
            json={"ids": merge_ids},
            timeout=30,
        )
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        logger.error("Failed to merge people into %s: %s", survivor_id, e)
        return False


def fetch_all_assets(person: dict) -> tuple[list[dict], int]:
    """Fetch all assets for a person with pagination.

    Returns (assets, total_raw) where assets is the list of valid dict items
    and total_raw is the raw item count across pages that had at least one valid
    dict. All-garbage pages (every item non-dict) stop pagination and are not
    counted. total_raw is a lower bound in two cases: a network error interrupts
    pagination (a warning is logged), or an all-garbage page terminates it early
    (a warning is logged and later pages are not fetched).
    """
    name = person.get("name", "Unknown")
    person_id = person.get("id")
    if not person_id:
        logger.error("Person dict missing 'id' field for %s — skipping asset fetch", name)
        return [], 0
    url = f"{Config.IMMICH_URL}/api/search/metadata"
    page_size = 1000

    logger.debug("Fetching assets for %s...", name)

    assets: list[dict] = []
    total_raw = 0  # raw item count across pages that yielded at least one valid dict
    for page in range(1, MAX_PAGES + 1):
        try:
            resp = requests.post(
                url,
                json={"personIds": [person_id], "size": page_size, "page": page},
                headers=get_headers(),
                timeout=30,
            )

            if not resp.ok:
                logger.error("Error fetching assets for %s (page %s): %s", name, page, resp.status_code)
                break

            page_assets = resp.json().get("assets", [])
            # Immich ≥2.x returns {"assets": {"items": [...]}};
            # earlier versions returned {"assets": [...]} directly.
            if isinstance(page_assets, dict):
                page_assets = page_assets.get("items", [])

            page_count = len(page_assets)  # raw count for termination check before filtering

            # Single pass: partition valid assets from unexpected non-dict items
            valid_assets, skipped_count = [], 0
            for item in page_assets:
                if isinstance(item, dict):
                    valid_assets.append(item)
                else:
                    skipped_count += 1
            if skipped_count:
                logger.warning("%s: skipping %s non-dict item(s) in page %s", name, skipped_count, page)

            if not valid_assets:
                if page_count > 0:
                    logger.warning(
                        "%s: page %s returned %s item(s) but none were valid dicts — stopping pagination",
                        name, page, page_count,
                    )
                break

            # Count page_count (not just valid items) so that non-dict items from a
            # transient schema issue on a mixed page don't cause MIN_FACE_COUNT to
            # skip a real person. Pages where every item is a non-dict are excluded —
            # they indicate a structural problem and break above without contributing.
            total_raw += page_count
            assets.extend(valid_assets)
            logger.debug("Fetched page %s, total: %s", page, len(assets))

            if page_count < page_size or len(assets) >= _MAX_ASSETS_PER_PERSON:
                break

        except (requests.RequestException, ValueError) as e:
            logger.error("Exception fetching assets for %s (page %s): %s", name, page, e)
            if page > 1:
                logger.warning(
                    "%s: pagination interrupted at page %s — total_raw=%s may undercount actual assets",
                    name, page, total_raw,
                )
            break

    return assets, total_raw


def fetch_face_data(asset_id: str, person_id: str | None = None) -> FaceData | None:
    """Fetch pre-computed face data (bbox, confidence) from Immich.

    Queries GET /api/faces?id={asset_id} to retrieve face detection results
    that Immich already computed using InsightFace Buffalo_L.

    Args:
        asset_id: The asset to get face data for
        person_id: Optional person ID to match the specific face

    Returns:
        FaceData with bbox and confidence, or None if unavailable
    """
    try:
        resp = requests.get(
            f"{Config.IMMICH_URL}/api/faces",
            params={"id": asset_id},
            headers=get_headers(),
            timeout=10,
        )

        if not resp.ok:
            logger.debug("Face data endpoint returned %s for %s", resp.status_code, asset_id)
            return None

        faces = resp.json()
        if not isinstance(faces, list) or not faces:
            return None

        # Match the target person if specified
        face = None
        if person_id:
            face = next(
                (f for f in faces if isinstance(f, dict) and (f.get("person") or {}).get("id") == person_id),
                None,
            )
        if face is None:
            face = faces[0] if isinstance(faces[0], dict) else None
        if face is None:
            return None

        bbox = (
            face.get("boundingBoxX1", 0),
            face.get("boundingBoxY1", 0),
            face.get("boundingBoxX2", 0),
            face.get("boundingBoxY2", 0),
        )

        score = face.get("score")
        return FaceData(
            bbox=bbox,
            confidence=score if score is not None else face.get("confidence"),
            image_width=face.get("imageWidth", 0),
            image_height=face.get("imageHeight", 0),
        )

    except requests.RequestException as e:
        logger.debug("Failed to fetch face data for %s: %s", asset_id, e)
        return None
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        logger.debug("Failed to parse face data for %s: %s", asset_id, e)
        return None


def fetch_full_image(asset_id: str, timeout: int = 60) -> Image.Image | None:
    """Fetch full-resolution image from Immich, falling back to preview thumbnail.

    The /original endpoint may return HEIC, RAW, or video files that PIL
    cannot open directly. In that case, we fall back to the JPEG thumbnail.
    """
    # Try original first
    try:
        resp = requests.get(
            f"{Config.IMMICH_URL}/api/assets/{asset_id}/original",
            headers=get_headers(),
            timeout=timeout,
        )
        if resp.ok:
            try:
                return ImageOps.exif_transpose(Image.open(BytesIO(resp.content)))
            except Exception:
                logger.debug("PIL can't open original for %s, falling back to preview", asset_id)
    except requests.RequestException:
        logger.debug("Original request failed for %s, falling back to preview", asset_id)

    # Fall back to preview thumbnail (always JPEG)
    try:
        resp = requests.get(
            f"{Config.IMMICH_URL}/api/assets/{asset_id}/thumbnail?size=preview&format=JPEG",
            headers=get_headers(),
            timeout=30,
        )
        if resp.ok:
            return ImageOps.exif_transpose(Image.open(BytesIO(resp.content)))
    except Exception as e:
        logger.error("Failed to fetch image %s: %s", asset_id, e)

    return None


def filter_recent_assets(assets: list[dict], years: int | None = None) -> list[dict]:
    """Filter assets to keep only those from the last N years."""
    years = years or Config.YEARS_FILTER
    cutoff = datetime.now(timezone.utc) - timedelta(days=365 * years)

    logger.debug("Filtering assets older than %s years (%s)", years, cutoff)

    recent, skipped = [], 0
    for asset in assets:
        created_at_str = asset.get("fileCreatedAt")
        if not created_at_str:
            continue

        try:
            # Handle ISO8601 with 'Z' suffix
            created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
            if created_at > cutoff:
                recent.append(asset)
            else:
                skipped += 1
        except ValueError:
            continue

    logger.debug("Retained %s assets (filtered %s old assets).", len(recent), skipped)
    return recent

