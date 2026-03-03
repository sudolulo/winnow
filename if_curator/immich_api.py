"""Immich API client for fetching people, assets, and face data."""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO

import numpy as np
import requests
from PIL import Image

from .config import Config, get_headers

logger = logging.getLogger(__name__)


@dataclass
class FaceData:
    """Pre-computed face data from Immich."""

    embedding: np.ndarray | None
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float | None
    image_width: int
    image_height: int


def get_people() -> list[dict]:
    """Fetch all people from Immich."""
    try:
        resp = requests.get(
            f"{Config.IMMICH_URL}/api/people",
            headers=get_headers(),
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("people", [])
    except requests.RequestException as e:
        logger.error(f"Failed to fetch people: {e}")
        return []


def fetch_all_assets(person: dict) -> list[dict]:
    """Fetch all assets for a person with pagination."""
    name = person.get("name", "Unknown")
    person_id = person["id"]
    url = f"{Config.IMMICH_URL}/api/search/metadata"
    page_size = 1000

    logger.info(f"Fetching assets for {name}...")

    assets = []
    for page in range(1, 1000):  # Safety limit
        try:
            resp = requests.post(
                url,
                json={"personIds": [person_id], "size": page_size, "page": page},
                headers=get_headers(),
                timeout=30,
            )

            if not resp.ok:
                logger.error(f"Error fetching assets for {name} (page {page}): {resp.status_code}")
                break

            page_assets = resp.json().get("assets", [])
            if isinstance(page_assets, dict):
                page_assets = page_assets.get("items", [])

            if not page_assets:
                break

            assets.extend(page_assets)
            logger.debug(f"Fetched page {page}, total: {len(assets)}")

            if len(page_assets) < page_size:
                break

        except requests.RequestException as e:
            logger.error(f"Exception fetching assets for {name}: {e}")
            break

    return assets


def fetch_face_data(asset_id: str, person_id: str | None = None) -> FaceData | None:
    """Fetch pre-computed face data (embedding, bbox, confidence) from Immich.

    Queries GET /api/faces?id={asset_id} to retrieve face detection results
    that Immich already computed using InsightFace Buffalo_L.

    Args:
        asset_id: The asset to get face data for
        person_id: Optional person ID to match the specific face

    Returns:
        FaceData with embedding, bbox, and confidence, or None if unavailable
    """
    try:
        resp = requests.get(
            f"{Config.IMMICH_URL}/api/faces",
            params={"id": asset_id},
            headers=get_headers(),
            timeout=10,
        )

        if not resp.ok:
            logger.debug(f"Face data endpoint returned {resp.status_code} for {asset_id}")
            return None

        faces = resp.json()
        if not faces:
            return None

        # Match the target person if specified
        face = None
        if person_id:
            face = next((f for f in faces if f.get("person", {}).get("id") == person_id), None)
        if face is None:
            face = faces[0]  # Fall back to first/largest face

        # Extract embedding if available
        embedding = None
        if "embedding" in face:
            embedding = np.array(face["embedding"], dtype=np.float32)

        # Extract bounding box
        bbox = (
            face.get("boundingBoxX1", 0),
            face.get("boundingBoxY1", 0),
            face.get("boundingBoxX2", 0),
            face.get("boundingBoxY2", 0),
        )

        return FaceData(
            embedding=embedding,
            bbox=bbox,
            confidence=face.get("score") or face.get("confidence"),
            image_width=face.get("imageWidth", 0),
            image_height=face.get("imageHeight", 0),
        )

    except requests.RequestException as e:
        logger.debug(f"Failed to fetch face data for {asset_id}: {e}")
        return None
    except (KeyError, TypeError, ValueError) as e:
        logger.debug(f"Failed to parse face data for {asset_id}: {e}")
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
                return Image.open(BytesIO(resp.content))
            except Exception:
                logger.debug(f"PIL can't open original for {asset_id}, falling back to preview")
    except requests.RequestException:
        logger.debug(f"Original request failed for {asset_id}, falling back to preview")

    # Fall back to preview thumbnail (always JPEG)
    try:
        resp = requests.get(
            f"{Config.IMMICH_URL}/api/assets/{asset_id}/thumbnail?size=preview&format=JPEG",
            headers=get_headers(),
            timeout=30,
        )
        if resp.ok:
            return Image.open(BytesIO(resp.content))
    except Exception as e:
        logger.error(f"Failed to fetch image {asset_id}: {e}")

    return None


def filter_recent_assets(assets: list[dict], years: int | None = None) -> list[dict]:
    """Filter assets to keep only those from the last N years."""
    years = years or Config.YEARS_FILTER
    cutoff = datetime.now(timezone.utc) - timedelta(days=365 * years)

    logger.debug(f"Filtering assets older than {years} years ({cutoff})")

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

    logger.info(f"Retained {len(recent)} assets (filtered {skipped} old assets).")
    return recent
