"""
spotify_client.py
-----------------
Handles all Spotify API authentication and raw data fetching.

Design decisions:
- Uses Authorization Code Flow (not Client Credentials) so we can access
  user-specific data: liked songs, top tracks, and create playlists.
- Fetches in batches to respect Spotify's API limits (max 50 items/request
  for tracks, max 100 audio features per request).
- Saves raw JSON to data/raw/ so we never need to re-fetch during development.
"""

import os
import json
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Scopes we need:
#   user-library-read     → access liked songs
#   user-top-read         → access top tracks
#   playlist-modify-public  → create/modify public playlists
#   playlist-modify-private → create/modify private playlists
SPOTIFY_SCOPES = " ".join([
    "user-library-read",
    "user-top-read",
    "playlist-modify-public",
    "playlist-modify-private",
])

# Spotify API hard limits
TRACKS_PER_REQUEST = 50       # max for saved tracks / top tracks endpoints
AUDIO_FEATURES_PER_REQUEST = 100  # max for audio_features endpoint


# ── Client ────────────────────────────────────────────────────────────────────
def get_spotify_client() -> spotipy.Spotify:
    """
    Authenticate and return an authorised Spotify client.

    Reads credentials from .env file. On first run this will open a browser
    window for OAuth consent. Spotipy caches the token in .cache so subsequent
    runs are silent.

    Returns
    -------
    spotipy.Spotify
        Authenticated Spotify client instance.
    """
    load_dotenv()

    client_id = os.getenv("SPOTIPY_CLIENT_ID")
    client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
    redirect_uri = os.getenv("SPOTIPY_REDIRECT_URI")

    if not all([client_id, client_secret, redirect_uri]):
        raise EnvironmentError(
            "Missing Spotify credentials. "
            "Copy .env.example to .env and fill in your values."
        )

    auth_manager = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=SPOTIFY_SCOPES,
        open_browser=True,
    )

    client = spotipy.Spotify(auth_manager=auth_manager, requests_timeout=10)
    logger.info("Spotify client authenticated successfully.")
    return client


# ── Fetchers ──────────────────────────────────────────────────────────────────
def fetch_liked_songs(client: spotipy.Spotify, limit: int = 500) -> list[dict]:
    """
    Fetch the current user's liked (saved) tracks.

    Parameters
    ----------
    client : spotipy.Spotify
        Authenticated Spotify client.
    limit : int
        Maximum number of tracks to fetch. Defaults to 500.

    Returns
    -------
    list[dict]
        List of raw track objects from the Spotify API.
    """
    tracks = []
    offset = 0

    logger.info(f"Fetching liked songs (up to {limit})...")

    while len(tracks) < limit:
        batch_size = min(TRACKS_PER_REQUEST, limit - len(tracks))
        response = client.current_user_saved_tracks(
            limit=batch_size,
            offset=offset
        )

        items = response.get("items", [])
        if not items:
            break

        tracks.extend(items)
        offset += len(items)
        logger.info(f"  Fetched {len(tracks)} liked songs so far...")

        # Respect rate limits
        if response.get("next") is None:
            break
        time.sleep(0.1)

    logger.info(f"Done. Total liked songs fetched: {len(tracks)}")
    return tracks


def fetch_top_tracks(
    client: spotipy.Spotify,
    time_range: str = "medium_term",
    limit: int = 100
) -> list[dict]:
    """
    Fetch the current user's top tracks.

    Parameters
    ----------
    client : spotipy.Spotify
        Authenticated Spotify client.
    time_range : str
        One of 'short_term' (4 weeks), 'medium_term' (6 months),
        'long_term' (all time). Defaults to 'medium_term'.
    limit : int
        Maximum number of tracks. Max 100. Defaults to 100.

    Returns
    -------
    list[dict]
        List of raw track objects from the Spotify API.
    """
    tracks = []
    offset = 0
    limit = min(limit, 100)  # API hard cap for top tracks

    logger.info(f"Fetching top tracks (time_range={time_range})...")

    while len(tracks) < limit:
        batch_size = min(TRACKS_PER_REQUEST, limit - len(tracks))
        response = client.current_user_top_tracks(
            limit=batch_size,
            offset=offset,
            time_range=time_range
        )

        items = response.get("items", [])
        if not items:
            break

        tracks.extend(items)
        offset += len(items)

        if response.get("next") is None:
            break
        time.sleep(0.1)

    logger.info(f"Done. Total top tracks fetched: {len(tracks)}")
    return tracks


def fetch_audio_features(
    client: spotipy.Spotify,
    track_ids: list[str]
) -> list[dict]:
    """
    Fetch audio features for a list of track IDs.

    Batches requests to stay within Spotify's 100-track limit per call.
    Filters out None responses (tracks Spotify has no features for).

    Parameters
    ----------
    client : spotipy.Spotify
        Authenticated Spotify client.
    track_ids : list[str]
        List of Spotify track IDs.

    Returns
    -------
    list[dict]
        List of audio feature objects. May be shorter than input if some
        tracks have no features available.
    """
    features = []
    total = len(track_ids)

    logger.info(f"Fetching audio features for {total} tracks...")

    for i in range(0, total, AUDIO_FEATURES_PER_REQUEST):
        batch = track_ids[i:i + AUDIO_FEATURES_PER_REQUEST]
        response = client.audio_features(batch)

        if response:
            valid = [f for f in response if f is not None]
            features.extend(valid)
            skipped = len(batch) - len(valid)
            if skipped:
                logger.warning(f"  {skipped} tracks had no audio features — skipped.")

        logger.info(f"  Processed {min(i + AUDIO_FEATURES_PER_REQUEST, total)}/{total}")
        time.sleep(0.1)

    logger.info(f"Done. Audio features retrieved: {len(features)}")
    return features


# ── Orchestrator ──────────────────────────────────────────────────────────────
def fetch_all_data(
    liked_limit: int = 500,
    top_limit: int = 100,
    save: bool = True
) -> dict:
    """
    Full data pull: liked songs + top tracks + audio features.

    Deduplicates tracks across both sources before fetching audio features
    to avoid redundant API calls.

    Parameters
    ----------
    liked_limit : int
        Max liked songs to fetch.
    top_limit : int
        Max top tracks to fetch.
    save : bool
        If True, saves raw data to data/raw/ as JSON files.

    Returns
    -------
    dict with keys:
        'tracks'        : deduplicated list of track metadata dicts
        'audio_features': list of audio feature dicts
    """
    client = get_spotify_client()

    # Fetch raw track lists
    liked_raw = fetch_liked_songs(client, limit=liked_limit)
    top_raw = fetch_top_tracks(client, limit=top_limit)

    # Normalise: liked songs are wrapped in {"added_at", "track": {...}}
    # Top tracks are the track object directly
    liked_tracks = [item["track"] for item in liked_raw if item.get("track")]
    top_tracks = top_raw  # already track objects

    # Deduplicate by track ID
    seen_ids = set()
    all_tracks = []
    for track in liked_tracks + top_tracks:
        tid = track.get("id")
        if tid and tid not in seen_ids:
            seen_ids.add(tid)
            all_tracks.append(track)

    logger.info(f"Total unique tracks after deduplication: {len(all_tracks)}")

    # Extract just the IDs for audio features call
    track_ids = [t["id"] for t in all_tracks]
    audio_features = fetch_audio_features(client, track_ids)

    result = {
        "tracks": all_tracks,
        "audio_features": audio_features,
    }

    if save:
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

        tracks_path = RAW_DATA_DIR / "tracks.json"
        features_path = RAW_DATA_DIR / "audio_features.json"

        with open(tracks_path, "w") as f:
            json.dump(all_tracks, f, indent=2)
        with open(features_path, "w") as f:
            json.dump(audio_features, f, indent=2)

        logger.info(f"Raw data saved to {RAW_DATA_DIR}")

    return result


def create_spotify_playlist(
    client: spotipy.Spotify,
    track_ids: list[str],
    name: str,
    description: str = "",
    public: bool = False
) -> str:
    """
    Create a new Spotify playlist and add tracks to it.

    Parameters
    ----------
    client : spotipy.Spotify
        Authenticated Spotify client.
    track_ids : list[str]
        Spotify track IDs to add.
    name : str
        Playlist name.
    description : str
        Playlist description shown in Spotify.
    public : bool
        Whether the playlist is public. Defaults to False.

    Returns
    -------
    str
        URL of the created playlist.
    """
    user_id = client.me()["id"]

    playlist = client.user_playlist_create(
        user=user_id,
        name=name,
        public=public,
        description=description
    )
    playlist_id = playlist["id"]

    # Add tracks in batches of 100 (Spotify limit)
    for i in range(0, len(track_ids), 100):
        batch = track_ids[i:i + 100]
        uris = [f"spotify:track:{tid}" for tid in batch]
        client.playlist_add_items(playlist_id, uris)
        time.sleep(0.1)

    playlist_url = playlist["external_urls"]["spotify"]
    logger.info(f"Playlist created: {playlist_url}")
    return playlist_url


# ── Entry point (quick test) ───────────────────────────────────────────────────
if __name__ == "__main__":
    data = fetch_all_data(liked_limit=200, top_limit=100)
    print(f"\nSummary:")
    print(f"  Unique tracks : {len(data['tracks'])}")
    print(f"  Audio features: {len(data['audio_features'])}")
