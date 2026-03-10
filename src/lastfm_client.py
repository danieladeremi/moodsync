"""
lastfm_client.py
----------------
Fetches track tags and artist metadata from the Last.fm API.

Last.fm provides crowd-sourced tags (e.g. "chill", "melancholic", "energetic")
that serve as our primary audio-character signal, replacing the Spotify
audio-features endpoint which is no longer available for new apps.

Design decisions:
- We use track.getTopTags as the primary signal: tags are weighted by how
  many users applied them, giving us a confidence score per tag.
- We fall back to artist.getTopTags if a track has fewer than 3 tags,
  since many tracks (especially newer ones) have sparse tag coverage.
- Tags are normalised to lowercase and deduplicated before storage.
- We rate-limit requests to ~2/second to stay within Last.fm's free tier
  limits (no hard rate limit documented, but 5 req/s is the safe ceiling).
- All raw responses are cached to data/raw/lastfm_tags.json so the
  expensive API loop only runs once during development.
"""

import os
import json
import time
import logging
from pathlib import Path

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
LASTFM_BASE_URL = "https://ws.audioscrobbler.com/2.0/"
RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Tags to use as binary/weighted ML features.
# These are the most consistently applied mood/genre tags on Last.fm.
# We score each track against these rather than one-hot encoding all tags
# (which would create thousands of sparse columns).
MOOD_TAGS = [
    # Energy level
    "energetic", "high energy", "upbeat", "uptempo",
    "calm", "relaxing", "mellow", "lo-fi", "ambient",
    # Mood
    "happy", "feel good", "feel-good", "uplifting",
    "sad", "melancholic", "melancholy", "dark", "depressing",
    "angry", "aggressive", "intense",
    # Activity
    "workout", "running", "gym",
    "study", "focus", "concentration", "sleep",
    "party", "dance", "dancing", "club",
    "chill", "chillout", "background",
    # Genre mood proxies
    "acoustic", "electronic", "instrumental",
    "indie", "pop", "hip-hop", "hip hop", "rap",
    "rock", "metal", "jazz", "classical", "soul", "rnb", "r&b",
]

# Canonical tag groups — we map raw tags to these clean names
TAG_ALIASES = {
    "feel good": "feel-good",
    "chillout": "chill",
    "chill out": "chill",
    "high energy": "energetic",
    "hip-hop": "hip hop",
    "rnb": "r&b",
    "hip hop": "hip hop",
    "lo fi": "lo-fi",
    "lofi": "lo-fi",
    "lo-fi hip hop": "lo-fi",
    "concentration": "focus",
    "studying": "study",
    "melancholy": "melancholic",
}


# ── API helpers ───────────────────────────────────────────────────────────────
def _get_api_key() -> str:
    load_dotenv()
    key = os.getenv("LASTFM_API_KEY")
    if not key:
        raise EnvironmentError(
            "Missing LASTFM_API_KEY. Add it to your .env file."
        )
    return key


def _call(params: dict, retries: int = 3) -> dict | None:
    """
    Make a single Last.fm API call with retry logic.

    Parameters
    ----------
    params : dict
        Query parameters (method, artist, track, etc.)
    retries : int
        Number of times to retry on failure.

    Returns
    -------
    dict | None
        Parsed JSON response, or None on failure.
    """
    api_key = _get_api_key()
    base_params = {
        "api_key": api_key,
        "format": "json",
        **params,
    }

    for attempt in range(retries):
        try:
            response = requests.get(
                LASTFM_BASE_URL,
                params=base_params,
                timeout=10,
            )
            if response.status_code == 200:
                data = response.json()
                if "error" in data:
                    logger.warning(f"Last.fm API error {data['error']}: {data.get('message')}")
                    return None
                return data
            else:
                logger.warning(f"HTTP {response.status_code} on attempt {attempt + 1}")
        except requests.RequestException as e:
            logger.warning(f"Request failed (attempt {attempt + 1}): {e}")

        time.sleep(1.0 * (attempt + 1))

    return None


# ── Tag fetchers ──────────────────────────────────────────────────────────────
def fetch_track_tags(artist: str, track: str) -> list[dict]:
    """
    Fetch top tags for a specific track.

    Parameters
    ----------
    artist : str
        Artist name.
    track : str
        Track name.

    Returns
    -------
    list[dict]
        List of {"name": str, "count": int} dicts, sorted by count desc.
        Empty list if no tags found.
    """
    data = _call({
        "method": "track.getTopTags",
        "artist": artist,
        "track": track,
        "autocorrect": 1,
    })

    if not data:
        return []

    tags = data.get("toptags", {}).get("tag", [])
    if isinstance(tags, dict):
        tags = [tags]

    return [
        {"name": t["name"].lower().strip(), "count": int(t["count"])}
        for t in tags
        if t.get("name") and int(t.get("count", 0)) > 0
    ]


def fetch_artist_tags(artist: str) -> list[dict]:
    """
    Fetch top tags for an artist (fallback when track has sparse tags).

    Parameters
    ----------
    artist : str
        Artist name.

    Returns
    -------
    list[dict]
        List of {"name": str, "count": int} dicts.
    """
    data = _call({
        "method": "artist.getTopTags",
        "artist": artist,
        "autocorrect": 1,
    })

    if not data:
        return []

    tags = data.get("toptags", {}).get("tag", [])
    if isinstance(tags, dict):
        tags = [tags]

    return [
        {"name": t["name"].lower().strip(), "count": int(t["count"])}
        for t in tags
        if t.get("name") and int(t.get("count", 0)) > 0
    ]


def fetch_track_info(artist: str, track: str) -> dict:
    """
    Fetch track info including global listener and playcount stats.

    Parameters
    ----------
    artist : str
    track : str

    Returns
    -------
    dict with keys: listeners (int), playcount (int). Defaults to 0.
    """
    data = _call({
        "method": "track.getInfo",
        "artist": artist,
        "track": track,
        "autocorrect": 1,
    })

    if not data:
        return {"listeners": 0, "playcount": 0}

    info = data.get("track", {})
    return {
        "listeners": int(info.get("listeners", 0)),
        "playcount": int(info.get("playcount", 0)),
    }


# ── Tag processing ────────────────────────────────────────────────────────────
def normalise_tags(tags: list[dict]) -> list[dict]:
    """
    Apply alias mapping and deduplicate tags.

    Parameters
    ----------
    tags : list[dict]
        Raw tag list from fetch_track_tags or fetch_artist_tags.

    Returns
    -------
    list[dict]
        Normalised, deduplicated list sorted by count desc.
    """
    seen = {}
    for tag in tags:
        name = TAG_ALIASES.get(tag["name"], tag["name"])
        if name in seen:
            seen[name] = max(seen[name], tag["count"])
        else:
            seen[name] = tag["count"]

    return sorted(
        [{"name": k, "count": v} for k, v in seen.items()],
        key=lambda x: x["count"],
        reverse=True,
    )


def tags_to_feature_vector(tags: list[dict]) -> dict[str, float]:
    """
    Convert a list of tags into a feature vector scored against MOOD_TAGS.

    Each MOOD_TAG becomes a feature. Its value is the normalised tag count
    (0–1) if the tag is present, 0 otherwise. Counts are normalised by
    the maximum count in the tag list to make scores comparable across tracks.

    Parameters
    ----------
    tags : list[dict]
        Normalised tag list (output of normalise_tags).

    Returns
    -------
    dict[str, float]
        Maps each MOOD_TAG → score in [0, 1].
    """
    tag_map = {t["name"]: t["count"] for t in tags}
    max_count = max(tag_map.values(), default=1)

    vector = {}
    for mood_tag in MOOD_TAGS:
        canonical = TAG_ALIASES.get(mood_tag, mood_tag)
        raw_count = tag_map.get(canonical, tag_map.get(mood_tag, 0))
        vector[f"tag_{mood_tag.replace(' ', '_').replace('-', '_').replace('&', 'and')}"] = (
            raw_count / max_count if max_count > 0 else 0.0
        )

    return vector


# ── Batch fetcher ─────────────────────────────────────────────────────────────
def fetch_all_lastfm_data(
    tracks: list[dict],
    save: bool = True,
    delay: float = 0.5,
) -> dict[str, dict]:
    """
    Fetch Last.fm tags and info for all tracks.

    For each track:
    1. Fetch track-level tags
    2. If fewer than 3 tags found, supplement with artist-level tags
    3. Fetch listener/playcount stats

    Parameters
    ----------
    tracks : list[dict]
        Raw Spotify track objects (must have 'name' and 'artists').
    save : bool
        If True, saves output to data/raw/lastfm_tags.json.
    delay : float
        Seconds to sleep between requests (rate limiting).

    Returns
    -------
    dict[str, dict]
        Maps track_id → {
            "tags": list[dict],
            "tag_vector": dict[str, float],
            "listeners": int,
            "playcount": int,
        }
    """
    results = {}
    total = len(tracks)

    # Load existing cache to avoid re-fetching
    cache_path = RAW_DATA_DIR / "lastfm_tags.json"
    if cache_path.exists():
        with open(cache_path) as f:
            results = json.load(f)
        logger.info(f"Loaded {len(results)} cached Last.fm records.")

    for i, track in enumerate(tracks):
        track_id = track.get("id")
        if not track_id or track_id in results:
            continue

        track_name = track.get("name", "")
        artists = track.get("artists", [])
        artist_name = artists[0]["name"] if artists else ""

        if not track_name or not artist_name:
            continue

        # Fetch track tags
        track_tags = fetch_track_tags(artist_name, track_name)
        time.sleep(delay)

        # Supplement with artist tags if sparse
        if len(track_tags) < 3:
            artist_tags = fetch_artist_tags(artist_name)
            time.sleep(delay)
            # Blend: track tags take priority, artist tags fill gaps
            existing_names = {t["name"] for t in track_tags}
            for at in artist_tags:
                if at["name"] not in existing_names:
                    track_tags.append({**at, "count": at["count"] // 2})

        normalised = normalise_tags(track_tags)
        tag_vector = tags_to_feature_vector(normalised)

        # Fetch listener/play stats
        info = fetch_track_info(artist_name, track_name)
        time.sleep(delay)

        results[track_id] = {
            "tags": normalised[:20],  # store top 20 tags
            "tag_vector": tag_vector,
            "listeners": info["listeners"],
            "playcount": info["playcount"],
        }

        if (i + 1) % 10 == 0:
            logger.info(f"  Last.fm progress: {i + 1}/{total} tracks processed")

            # Save incrementally every 10 tracks to preserve progress
            if save:
                RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "w") as f:
                    json.dump(results, f, indent=2)

    if save:
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Last.fm data saved to {cache_path}")

    logger.info(f"Last.fm fetch complete. {len(results)} tracks with data.")
    return results


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import json as _json
    sys.path.insert(0, str(Path(__file__).parent))

    tracks_path = RAW_DATA_DIR / "tracks.json"
    if not tracks_path.exists():
        print("ERROR: data/raw/tracks.json not found. Run spotify_client.py first.")
        sys.exit(1)

    with open(tracks_path) as f:
        tracks = _json.load(f)

    print(f"Fetching Last.fm data for {len(tracks)} tracks...")
    print("This will take several minutes. Progress saves every 10 tracks.")

    results = fetch_all_lastfm_data(tracks, save=True, delay=0.5)

    print(f"Done. {len(results)} tracks processed.")
    print("You can now run: python src/features.py")