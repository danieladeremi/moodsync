"""
features.py
-----------
Merges track metadata with audio features into a clean, analysis-ready
DataFrame.

Design decisions:
- We keep both the audio features (ML inputs) and track metadata (name,
  artist, album art) in one DataFrame so downstream steps never need to
  re-join.
- We engineer two composite features (energy_valence, danceability_tempo_norm)
  that tend to produce better cluster separation than raw features alone.
  These are kept alongside the originals so the model can decide.
- We do NOT scale here. Scaling is the job of preprocess.py. This module
  only extracts and engineers.
"""

import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# ── The 9 core Spotify audio features used as ML inputs ───────────────────────
AUDIO_FEATURE_COLS = [
    "energy",           # Intensity and activity (0–1)
    "valence",          # Musical positivity / happiness (0–1)
    "danceability",     # Rhythm suitability for dancing (0–1)
    "tempo",            # BPM — NOT on 0–1 scale, needs StandardScaler
    "acousticness",     # Acoustic vs. electronic signal (0–1)
    "instrumentalness", # Absence of vocals (0–1)
    "liveness",         # Audience presence in recording (0–1)
    "loudness",         # Average dB — negative values, needs StandardScaler
    "speechiness",      # Proportion of spoken words (0–1)
]

# Features that are NOT on the 0–1 scale and will need StandardScaler
NEEDS_SCALING = ["tempo", "loudness"]

# Metadata columns we want to carry alongside features
METADATA_COLS = [
    "id",
    "name",
    "artist",
    "album",
    "album_art_url",
    "duration_ms",
    "popularity",
    "uri",
]


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_raw_data(
    tracks_path: Path | None = None,
    features_path: Path | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Load raw JSON files saved by spotify_client.fetch_all_data().

    Parameters
    ----------
    tracks_path : Path, optional
        Override default path to tracks.json.
    features_path : Path, optional
        Override default path to audio_features.json.

    Returns
    -------
    tuple[list[dict], list[dict]]
        (tracks, audio_features) as lists of dicts.
    """
    tracks_path = tracks_path or RAW_DATA_DIR / "tracks.json"
    features_path = features_path or RAW_DATA_DIR / "audio_features.json"

    with open(tracks_path) as f:
        tracks = json.load(f)
    with open(features_path) as f:
        audio_features = json.load(f)

    logger.info(f"Loaded {len(tracks)} tracks and {len(audio_features)} feature records.")
    return tracks, audio_features


# ── Parsers ───────────────────────────────────────────────────────────────────
def parse_tracks(tracks: list[dict]) -> pd.DataFrame:
    """
    Flatten raw track objects into a tidy metadata DataFrame.

    Extracts the first artist name and the largest album image URL.

    Parameters
    ----------
    tracks : list[dict]
        Raw track objects from the Spotify API.

    Returns
    -------
    pd.DataFrame
        One row per track with columns matching METADATA_COLS.
    """
    rows = []
    for t in tracks:
        # Some tracks may be malformed — skip rather than crash
        if not t or not t.get("id"):
            continue

        artists = t.get("artists", [])
        artist_name = artists[0]["name"] if artists else "Unknown"

        album = t.get("album", {})
        images = album.get("images", [])
        # Images are sorted largest → smallest; take the first (largest)
        art_url = images[0]["url"] if images else None

        rows.append({
            "id": t["id"],
            "name": t.get("name", ""),
            "artist": artist_name,
            "album": album.get("name", ""),
            "album_art_url": art_url,
            "duration_ms": t.get("duration_ms"),
            "popularity": t.get("popularity"),
            "uri": t.get("uri", ""),
        })

    df = pd.DataFrame(rows)
    logger.info(f"Parsed {len(df)} track metadata rows.")
    return df


def parse_audio_features(audio_features: list[dict]) -> pd.DataFrame:
    """
    Flatten raw audio feature objects into a tidy DataFrame.

    Keeps only the 9 core ML feature columns plus track ID for joining.

    Parameters
    ----------
    audio_features : list[dict]
        Raw audio feature objects from the Spotify API.

    Returns
    -------
    pd.DataFrame
        One row per track with columns: ['id'] + AUDIO_FEATURE_COLS.
    """
    rows = []
    for f in audio_features:
        if not f or not f.get("id"):
            continue
        row = {"id": f["id"]}
        for col in AUDIO_FEATURE_COLS:
            row[col] = f.get(col)
        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(f"Parsed {len(df)} audio feature rows.")
    return df


# ── Feature engineering ───────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add composite features that improve cluster separation.

    New features:
    - energy_valence : product of energy × valence. High = euphoric,
      Low = dark/calm. Captures the mood quadrant more directly than
      either feature alone.
    - danceability_energy : product of danceability × energy. High = workout,
      Low = ambient/sleep.
    - tempo_norm : tempo normalised to [0, 1] using the observed range in
      this dataset. Lets us combine tempo with 0–1 features without scaling
      side effects in engineered products.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain all AUDIO_FEATURE_COLS.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with 3 new columns appended.
    """
    df = df.copy()

    df["energy_valence"] = df["energy"] * df["valence"]
    df["danceability_energy"] = df["danceability"] * df["energy"]

    # Normalise tempo to [0, 1] using dataset min/max
    t_min, t_max = df["tempo"].min(), df["tempo"].max()
    if t_max > t_min:
        df["tempo_norm"] = (df["tempo"] - t_min) / (t_max - t_min)
    else:
        df["tempo_norm"] = 0.5

    logger.info("Engineered 3 composite features: energy_valence, danceability_energy, tempo_norm.")
    return df


# ── Main builder ──────────────────────────────────────────────────────────────
def build_feature_dataframe(
    tracks: list[dict] | None = None,
    audio_features: list[dict] | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Full pipeline: load → parse → merge → engineer → return DataFrame.

    If tracks/audio_features are not passed, loads from data/raw/.

    Parameters
    ----------
    tracks : list[dict], optional
        Raw track objects. If None, loads from disk.
    audio_features : list[dict], optional
        Raw audio feature objects. If None, loads from disk.
    save : bool
        If True, saves the merged DataFrame to data/processed/features.csv.

    Returns
    -------
    pd.DataFrame
        One row per track with metadata + audio features + engineered features.
    """
    if tracks is None or audio_features is None:
        tracks, audio_features = load_raw_data()

    tracks_df = parse_tracks(tracks)
    features_df = parse_audio_features(audio_features)

    # Inner join: only keep tracks that have both metadata AND audio features
    df = tracks_df.merge(features_df, on="id", how="inner")
    logger.info(f"Merged DataFrame: {len(df)} rows (inner join on track ID).")

    # Drop rows with any null in the audio feature columns
    before = len(df)
    df = df.dropna(subset=AUDIO_FEATURE_COLS)
    dropped = before - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped} rows with null audio features.")

    # Engineer composite features
    df = engineer_features(df)

    # Reset index cleanly
    df = df.reset_index(drop=True)

    if save:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PROCESSED_DATA_DIR / "features.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"Feature DataFrame saved to {out_path}")

    return df


# ── Convenience getters ───────────────────────────────────────────────────────
def get_ml_feature_cols() -> list[str]:
    """
    Return the full list of feature column names used as ML model inputs.
    Includes original audio features + engineered features (excluding tempo
    and loudness raw — those are scaled in preprocess.py).
    """
    engineered = ["energy_valence", "danceability_energy", "tempo_norm"]
    return AUDIO_FEATURE_COLS + engineered


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = build_feature_dataframe()
    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nSample row:\n{df.iloc[0]}")
