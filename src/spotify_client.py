"""
spotify_client.py
-----------------
Handles all Spotify API authentication and raw data fetching.

Design decisions:
- Uses Authorization Code Flow so we can access user-specific data:
  liked songs, top tracks, and create playlists.
- Spotify deprecated the following endpoints for new apps (Nov 27 2024):
    audio-features, audio-analysis, artists (batch), recommendations,
    related-artists, featured-playlists, category-playlists.
  We use NONE of these.
- Genre data: artist genres are NOT available via the artists batch endpoint
  for new apps. However, genres ARE embedded inside the artist objects that
  come back within track responses from saved_tracks and top_tracks — we
  extract them directly from there.
- Endpoints we DO use (all still available to new apps):
    current_user_saved_tracks  → liked songs
    current_user_top_tracks    → top tracks
    user_playlist_create       → create playlist
    playlist_add_items         → add tracks to playlist
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

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

SPOTIFY_SCOPES = " ".join([
    "user-library-read",
    "user-top-read",
    "playlist-modify-public",
    "playlist-modify-private",
])

TRACKS_PER_REQUEST = 50


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
    limit : int
        Maximum number of tracks to fetch.

    Returns
    -------
    list[dict]
        List of saved track wrapper objects {"added_at", "track": {...}}.
        The track object contains artists with genre data embedded.
    """
    tracks = []
    offset = 0

    logger.info(f"Fetching liked songs (up to {limit})...")

    while len(tracks) < limit:
        batch_size = min(TRACKS_PER_REQUEST, limit - len(tracks))
        response = client.current_user_saved_tracks(limit=batch_size, offset=offset)

        items = response.get("items", [])
        if not items:
            break

        tracks.extend(items)
        offset += len(items)
        logger.info(f"  Fetched {len(tracks)} liked songs so far...")

        if response.get("next") is None:
            break
        time.sleep(0.1)

    logger.info(f"Done. Total liked songs fetched: {len(tracks)}")
    return tracks


def fetch_top_tracks(
    client: spotipy.Spotify,
    time_range: str = "medium_term",
    limit: int = 100,
) -> list[dict]:
    """
    Fetch the current user's top tracks.

    Parameters
    ----------
    client : spotipy.Spotify
    time_range : str
        One of 'short_term' (4 weeks), 'medium_term' (6 months),
        'long_term' (all time).
    limit : int
        Maximum number of tracks. Max 100.

    Returns
    -------
    list[dict]
        List of track objects.
    """
    tracks = []
    offset = 0
    limit = min(limit, 100)

    logger.info(f"Fetching top tracks (time_range={time_range})...")

    while len(tracks) < limit:
        batch_size = min(TRACKS_PER_REQUEST, limit - len(tracks))
        response = client.current_user_top_tracks(
            limit=batch_size,
            offset=offset,
            time_range=time_range,
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


# ── Orchestrator ──────────────────────────────────────────────────────────────
def fetch_all_data(
    liked_limit: int = 500,
    top_limit: int = 100,
    save: bool = True,
) -> dict:
    """
    Full data pull: liked songs + top tracks.

    Genre data is extracted directly from the artist objects embedded in
    each track response — no separate artists endpoint call needed.

    Parameters
    ----------
    liked_limit : int
    top_limit : int
    save : bool
        If True, saves raw data to data/raw/ as JSON files.

    Returns
    -------
    dict with keys:
        'tracks' : deduplicated list of track objects
    """
    client = get_spotify_client()

    liked_raw = fetch_liked_songs(client, limit=liked_limit)
    top_raw = fetch_top_tracks(client, limit=top_limit)

    # Liked songs are wrapped: {"added_at": ..., "track": {...}}
    liked_tracks = [item["track"] for item in liked_raw if item.get("track")]
    top_tracks = top_raw

    # Deduplicate by track ID
    seen_ids = set()
    all_tracks = []
    for track in liked_tracks + top_tracks:
        tid = track.get("id")
        if tid and tid not in seen_ids:
            seen_ids.add(tid)
            all_tracks.append(track)

    logger.info(f"Total unique tracks after deduplication: {len(all_tracks)}")

    # Log how many tracks have genre data in their artist objects
    with_genres = sum(
        1 for t in all_tracks
        if any(a.get("genres") for a in t.get("artists", []))
    )
    logger.info(f"Tracks with embedded genre data: {with_genres}/{len(all_tracks)}")

    result = {"tracks": all_tracks}

    if save:
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(RAW_DATA_DIR / "tracks.json", "w") as f:
            json.dump(all_tracks, f, indent=2)
        logger.info(f"Raw data saved to {RAW_DATA_DIR}")

    return result


def create_spotify_playlist(
    client: spotipy.Spotify,
    track_ids: list[str],
    name: str,
    description: str = "",
    public: bool = False,
) -> str:
    """
    Create a new Spotify playlist and add tracks to it.

    Parameters
    ----------
    client : spotipy.Spotify
    track_ids : list[str]
    name : str
    description : str
    public : bool

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
        description=description,
    )
    playlist_id = playlist["id"]

    for i in range(0, len(track_ids), 100):
        batch = track_ids[i:i + 100]
        uris = [f"spotify:track:{tid}" for tid in batch]
        client.playlist_add_items(playlist_id, uris)
        time.sleep(0.1)

    playlist_url = playlist["external_urls"]["spotify"]
    logger.info(f"Playlist created: {playlist_url}")
    return playlist_url


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = fetch_all_data(liked_limit=200, top_limit=100)
    print(f"\nSummary:")
    print(f"  Unique tracks: {len(data['tracks'])}")

    # Show a sample of genre data from the first few tracks
    print(f"\nSample genre data:")
    for track in data["tracks"][:5]:
        name = track.get("name", "")
        artists = track.get("artists", [])
        artist_name = artists[0].get("name", "") if artists else ""
        genres = artists[0].get("genres", []) if artists else []
        print(f"  {name} — {artist_name}: {genres[:3]}")