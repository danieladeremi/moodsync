"""
features.py
-----------
Builds the ML feature DataFrame from Spotify track metadata and Last.fm tags.

Feature sources:

  Spotify metadata (per track):
    - popularity       : Spotify's 0–100 popularity score
    - duration_ms      : track length in milliseconds
    - explicit         : binary 0/1
    - release_year     : extracted from album release date

  Last.fm tag vectors (weighted 0–1):
    - tag_energetic, tag_chill, tag_sad, tag_happy, etc.
    - Crowd-sourced mood/activity tags normalised by tag count
    - Primary mood/character signal for clustering

  Last.fm popularity signals:
    - listeners_log    : log10(listeners + 1)
    - playcount_log    : log10(playcount + 1)

  Engineered features:
    - mood_score       : positive minus negative tag proxy (valence-like)
    - energy_proxy     : energetic/workout/dance tag combination
    - release_recency  : normalised release year within library

Note on Spotify genres:
  Spotify's artist genres are not available to new apps. The simplified
  artist objects embedded in track responses do not include genre data,
  and the artists batch endpoint (/v1/artists) is restricted for apps
  created after November 27, 2024. Last.fm tags serve as a richer,
  crowd-sourced replacement.

Design decisions:
- Log scaling on listener/playcount: raw values span several orders of
  magnitude and would dominate distance calculations unscaled.
- We do NOT scale here. Scaling is the job of preprocess.py.
"""

import json
import logging
import math
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# ── Config ────────────────────────────────────────────────────────────────────
METADATA_COLS = [
    "id", "name", "artist", "album", "album_art_url",
    "duration_ms", "popularity", "uri",
]

# Columns that need StandardScaler (not on 0–1 scale)
NEEDS_SCALING = ["popularity", "duration_ms", "listeners_log", "playcount_log"]


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_raw_data() -> tuple[list[dict], dict[str, dict]]:
    """
    Load raw JSON files produced by spotify_client and lastfm_client.

    Returns
    -------
    tuple: (tracks, lastfm_data)
    """
    with open(RAW_DATA_DIR / "tracks.json") as f:
        tracks = json.load(f)

    lastfm_path = RAW_DATA_DIR / "lastfm_tags.json"
    if lastfm_path.exists():
        with open(lastfm_path) as f:
            lastfm_data = json.load(f)
    else:
        logger.warning("lastfm_tags.json not found. Run lastfm_client.py first.")
        lastfm_data = {}

    logger.info(f"Loaded {len(tracks)} tracks, {len(lastfm_data)} Last.fm records.")
    return tracks, lastfm_data


# ── Parsers ───────────────────────────────────────────────────────────────────
def parse_tracks(tracks: list[dict]) -> pd.DataFrame:
    """
    Flatten raw Spotify track objects into a tidy metadata DataFrame.
    """
    rows = []
    for t in tracks:
        if not t or not t.get("id"):
            continue

        artists = t.get("artists", [])
        artist_name = artists[0].get("name", "Unknown") if artists else "Unknown"

        album = t.get("album", {})
        images = album.get("images", [])
        art_url = images[0]["url"] if images else None

        release_date = album.get("release_date", "")
        try:
            release_year = int(release_date[:4]) if release_date else 2000
        except ValueError:
            release_year = 2000

        rows.append({
            "id": t["id"],
            "name": t.get("name", ""),
            "artist": artist_name,
            "album": album.get("name", ""),
            "album_art_url": art_url,
            "duration_ms": t.get("duration_ms", 0),
            "popularity": t.get("popularity", 0),
            "explicit": int(t.get("explicit", False)),
            "release_year": release_year,
            "uri": t.get("uri", ""),
        })

    df = pd.DataFrame(rows)
    logger.info(f"Parsed {len(df)} track metadata rows.")
    return df


def build_lastfm_features(
    df: pd.DataFrame,
    lastfm_data: dict[str, dict],
) -> pd.DataFrame:
    """
    Add Last.fm tag vector and popularity features to the DataFrame.

    For tracks with no Last.fm data, tag features default to 0
    and listener/playcount to log10(1) = 0.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'id' column.
    lastfm_data : dict
        Maps track_id → {tag_vector, listeners, playcount}.

    Returns
    -------
    pd.DataFrame
        With tag_* columns and listeners_log, playcount_log added.
    """
    df = df.copy()

    # Get full set of tag column names from any record
    tag_cols = []
    for record in lastfm_data.values():
        if record.get("tag_vector"):
            tag_cols = list(record["tag_vector"].keys())
            break

    if not tag_cols:
        logger.warning(
            "No tag vectors found in Last.fm data. "
            "Run python src/lastfm_client.py first."
        )
        return df

    tag_rows = []
    for _, row in df.iterrows():
        record = lastfm_data.get(row["id"], {})
        tag_vector = record.get("tag_vector", {})
        listeners = record.get("listeners", 0)
        playcount = record.get("playcount", 0)

        feature_row = {col: tag_vector.get(col, 0.0) for col in tag_cols}
        feature_row["listeners_log"] = math.log10(listeners + 1)
        feature_row["playcount_log"] = math.log10(playcount + 1)
        tag_rows.append(feature_row)

    tag_df = pd.DataFrame(tag_rows, index=df.index)
    df = pd.concat([df, tag_df], axis=1)

    coverage = (df["listeners_log"] > 0).sum()
    logger.info(
        f"Added {len(tag_cols)} Last.fm tag features. "
        f"Coverage: {coverage}/{len(df)} tracks ({coverage/len(df)*100:.0f}%)."
    )
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered composite features.

    - mood_score       : positive minus negative tag proxy (valence-like)
    - energy_proxy     : energetic/workout/dance tag combination
    - release_recency  : normalised release year within library
    """
    df = df.copy()

    def get_col(name):
        col = f"tag_{name}"
        return df[col] if col in df.columns else pd.Series(0.0, index=df.index)

    positive = (
        get_col("happy") + get_col("feel_good") +
        get_col("uplifting") + get_col("upbeat")
    )
    negative = (
        get_col("sad") + get_col("melancholic") +
        get_col("dark") + get_col("depressing")
    )
    df["mood_score"] = ((positive - negative) / 4.0).clip(-1, 1)

    df["energy_proxy"] = (
        get_col("energetic") + get_col("workout") +
        get_col("dance") + get_col("party")
    ).clip(0, 4) / 4.0

    yr_min = df["release_year"].min()
    yr_max = df["release_year"].max()
    if yr_max > yr_min:
        df["release_recency"] = (df["release_year"] - yr_min) / (yr_max - yr_min)
    else:
        df["release_recency"] = 0.5

    logger.info("Engineered 3 composite features: mood_score, energy_proxy, release_recency.")
    return df


# ── Main builder ──────────────────────────────────────────────────────────────
def build_feature_dataframe(
    tracks: list[dict] | None = None,
    lastfm_data: dict | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Full pipeline: load → parse → Last.fm features → engineer → return.
    """
    if tracks is None or lastfm_data is None:
        tracks, lastfm_data = load_raw_data()

    df = parse_tracks(tracks)
    df = build_lastfm_features(df, lastfm_data)
    df = engineer_features(df)

    df = df.dropna(subset=["id", "name"])
    df = df.reset_index(drop=True)

    if save:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PROCESSED_DATA_DIR / "features.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"Feature DataFrame saved to {out_path} — shape: {df.shape}")

    return df


def get_ml_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return ML feature columns, excluding metadata and intermediate columns."""
    exclude = set(METADATA_COLS + ["explicit", "release_year"])
    return [c for c in df.columns if c not in exclude]


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = build_feature_dataframe()
    ml_cols = get_ml_feature_cols(df)
    print(f"\nDataFrame shape  : {df.shape}")
    print(f"ML feature count : {len(ml_cols)}")
    print(f"\nSample features  : {ml_cols[:10]}")