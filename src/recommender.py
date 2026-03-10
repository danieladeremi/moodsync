"""
recommender.py
--------------
Maps a user's mood input to a cluster and returns a playlist of tracks.

Design decisions:
- We use centroid distance (cosine similarity) to match a mood string to
  a cluster — not collaborative filtering. This means the system works with
  only one user's data and requires no other users' history.
- Mood input is matched to our MOOD_ARCHETYPES first, then the archetype
  is compared against cluster centroids. This gives us a natural language
  interface without needing an NLP model.
- Tracks within a matched cluster are ranked by their distance to the centroid
  (closest first) so the most "archetypal" tracks appear at the top.
- An optional diversity parameter shuffles the ranking slightly to avoid
  returning the same playlist every time.
"""

import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from cluster import MOOD_ARCHETYPES
from features import AUDIO_FEATURE_COLS
from preprocess import transform_single_track, get_model_input_cols

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PROCESSED_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_model_artifacts() -> tuple:
    """
    Load the trained K-Means model and fitted scaler from disk.

    Returns
    -------
    tuple: (KMeans, StandardScaler)
    """
    model_path = MODELS_DIR / "kmeans_model.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            "No trained model found. Run cluster.run_clustering_pipeline() first."
        )
    if not scaler_path.exists():
        raise FileNotFoundError(
            "No fitted scaler found. Run preprocess.fit_and_scale() first."
        )

    km = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return km, scaler


def load_clustered_tracks() -> pd.DataFrame:
    """
    Load the clustered tracks DataFrame saved by cluster.py.

    Returns
    -------
    pd.DataFrame
    """
    path = PROCESSED_DATA_DIR / "clustered_tracks.csv"
    if not path.exists():
        raise FileNotFoundError(
            "No clustered tracks found. Run cluster.run_clustering_pipeline() first."
        )
    return pd.read_csv(path)


# ── Mood matching ─────────────────────────────────────────────────────────────
def match_mood_to_archetype(mood_query: str) -> tuple[str, dict]:
    """
    Match a free-text mood query to the closest named mood archetype.

    Uses simple keyword matching first, then falls back to returning
    the archetype whose name has the most overlapping words with the query.

    Parameters
    ----------
    mood_query : str
        User's mood input, e.g. "workout", "sad", "studying", "party".

    Returns
    -------
    tuple[str, dict]
        (archetype_name, archetype_feature_dict)
    """
    query_lower = mood_query.lower()

    # Keyword → archetype name mapping for common inputs
    keyword_map = {
        "workout": "Energetic / Workout",
        "exercise": "Energetic / Workout",
        "gym": "Energetic / Workout",
        "run": "Energetic / Workout",
        "running": "Energetic / Workout",
        "happy": "Happy / Feel-Good",
        "feel good": "Happy / Feel-Good",
        "feel-good": "Happy / Feel-Good",
        "good vibes": "Happy / Feel-Good",
        "upbeat": "Happy / Feel-Good",
        "chill": "Chill / Relaxed",
        "relax": "Chill / Relaxed",
        "relaxed": "Chill / Relaxed",
        "calm": "Chill / Relaxed",
        "easy": "Chill / Relaxed",
        "focus": "Focus / Deep Work",
        "study": "Focus / Deep Work",
        "studying": "Focus / Deep Work",
        "work": "Focus / Deep Work",
        "concentrate": "Focus / Deep Work",
        "deep work": "Focus / Deep Work",
        "sad": "Melancholic / Sad",
        "melancholic": "Melancholic / Sad",
        "melancholy": "Melancholic / Sad",
        "cry": "Melancholic / Sad",
        "heartbreak": "Melancholic / Sad",
        "late night": "Late Night / Dark",
        "night": "Late Night / Dark",
        "dark": "Late Night / Dark",
        "moody": "Late Night / Dark",
        "midnight": "Late Night / Dark",
        "party": "Party / Dance",
        "dance": "Party / Dance",
        "dancing": "Party / Dance",
        "club": "Party / Dance",
        "hype": "Party / Dance",
    }

    for keyword, archetype_name in keyword_map.items():
        if keyword in query_lower:
            logger.info(f"Mood '{mood_query}' matched to archetype '{archetype_name}' via keyword.")
            return archetype_name, MOOD_ARCHETYPES[archetype_name]

    # Fallback: word overlap scoring
    query_words = set(query_lower.split())
    best_name = None
    best_overlap = -1

    for name in MOOD_ARCHETYPES:
        name_words = set(name.lower().replace("/", " ").split())
        overlap = len(query_words & name_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_name = name

    # Final fallback: default to Chill
    if best_name is None or best_overlap == 0:
        best_name = "Chill / Relaxed"
        logger.warning(f"No match found for '{mood_query}'. Defaulting to 'Chill / Relaxed'.")
    else:
        logger.info(f"Mood '{mood_query}' matched to archetype '{best_name}' via word overlap.")

    return best_name, MOOD_ARCHETYPES[best_name]


def find_best_cluster_for_archetype(
    archetype_features: dict,
    km,
    scaler,
    feature_cols: list[str],
) -> int:
    """
    Find the cluster whose centroid is closest (cosine similarity) to the
    given archetype feature vector.

    Parameters
    ----------
    archetype_features : dict
        Feature dict from MOOD_ARCHETYPES.
    km : KMeans
        Fitted K-Means model.
    scaler : StandardScaler
        Fitted scaler.
    feature_cols : list[str]
        Ordered feature column names used in the model.

    Returns
    -------
    int
        Cluster ID of the best matching cluster.
    """
    # Transform archetype into the same scaled feature space as the model
    archetype_scaled = transform_single_track(archetype_features, scaler)

    centroids = km.cluster_centers_
    best_cluster = None
    best_sim = -np.inf

    for cluster_id, centroid in enumerate(centroids):
        a = archetype_scaled[0]
        b = centroid
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            sim = 0.0
        else:
            sim = np.dot(a, b) / (a_norm * b_norm)

        if sim > best_sim:
            best_sim = sim
            best_cluster = cluster_id

    logger.info(f"Best cluster for archetype: {best_cluster} (cosine sim: {best_sim:.4f})")
    return best_cluster


# ── Playlist builder ──────────────────────────────────────────────────────────
def get_playlist(
    mood_query: str,
    n_tracks: int = 20,
    diversity: float = 0.2,
    km=None,
    scaler=None,
    df_clustered: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Main entry point. Given a mood string, return a DataFrame of recommended
    tracks.

    Parameters
    ----------
    mood_query : str
        Natural language mood input from the user.
    n_tracks : int
        Number of tracks to return.
    diversity : float
        Value between 0.0 and 1.0. Higher values introduce more randomness
        into track selection (less centroid-biased). 0.0 = always return
        the same tracks; 1.0 = fully random within cluster.
    km : KMeans, optional
        Pre-loaded model (avoids re-loading from disk).
    scaler : StandardScaler, optional
        Pre-loaded scaler (avoids re-loading from disk).
    df_clustered : pd.DataFrame, optional
        Pre-loaded clustered tracks (avoids re-loading from disk).

    Returns
    -------
    pd.DataFrame
        Columns: name, artist, album, album_art_url, uri, mood_label,
                 cluster_id, centroid_distance
    """
    # Load artifacts if not provided
    if km is None or scaler is None:
        km, scaler = load_model_artifacts()
    if df_clustered is None:
        df_clustered = load_clustered_tracks()

    feature_cols = get_model_input_cols()

    # Match mood to archetype → cluster
    archetype_name, archetype_features = match_mood_to_archetype(mood_query)
    target_cluster = find_best_cluster_for_archetype(
        archetype_features, km, scaler, feature_cols
    )

    # Filter to tracks in the target cluster
    cluster_tracks = df_clustered[df_clustered["cluster_id"] == target_cluster].copy()

    if cluster_tracks.empty:
        logger.warning(f"Cluster {target_cluster} is empty. Returning empty playlist.")
        return pd.DataFrame()

    # Score tracks by distance to centroid (closer = more archetypal for this mood)
    centroid = km.cluster_centers_[target_cluster]

    # Get the feature columns that exist in the DataFrame
    available_feature_cols = [c for c in feature_cols if c in cluster_tracks.columns]

    if available_feature_cols:
        track_vectors = cluster_tracks[available_feature_cols].fillna(0).values
        distances = np.linalg.norm(track_vectors - centroid[:len(available_feature_cols)], axis=1)
        cluster_tracks["centroid_distance"] = distances

        # Sort by distance (ascending = closest to centroid first)
        cluster_tracks = cluster_tracks.sort_values("centroid_distance")
    else:
        cluster_tracks["centroid_distance"] = 0.0

    # Apply diversity: optionally shuffle a portion of the ranking
    if diversity > 0 and len(cluster_tracks) > n_tracks:
        top_n = max(n_tracks, int(len(cluster_tracks) * (1 - diversity)))
        top_pool = cluster_tracks.iloc[:top_n]
        playlist = top_pool.sample(n=min(n_tracks, len(top_pool)), random_state=None)
    else:
        playlist = cluster_tracks.head(n_tracks)

    # Select output columns
    output_cols = [
        "name", "artist", "album", "album_art_url", "uri",
        "mood_label", "cluster_id", "centroid_distance",
        "energy", "valence", "danceability", "tempo"
    ]
    output_cols = [c for c in output_cols if c in playlist.columns]
    playlist = playlist[output_cols].reset_index(drop=True)

    logger.info(
        f"Playlist generated: {len(playlist)} tracks for mood '{mood_query}' "
        f"→ archetype '{archetype_name}' → cluster {target_cluster}"
    )
    return playlist


def get_available_moods() -> list[str]:
    """Return the list of named mood archetypes available in the system."""
    return list(MOOD_ARCHETYPES.keys())


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    playlist = get_playlist("workout", n_tracks=10)
    if not playlist.empty:
        print(f"\nPlaylist ({len(playlist)} tracks):\n")
        for _, row in playlist.iterrows():
            print(f"  {row['name']} — {row['artist']}  [{row.get('mood_label', '')}]")
    else:
        print("No tracks found. Have you run the clustering pipeline yet?")
