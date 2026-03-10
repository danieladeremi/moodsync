"""
preprocess.py
-------------
Scales and prepares the feature DataFrame for clustering.

Design decisions:
- We apply StandardScaler ONLY to columns in NEEDS_SCALING (popularity,
  duration_ms, release_year, listeners_log, playcount_log). These are on
  different numeric scales and would otherwise dominate distance calculations.
- Genre columns (binary 0/1) and Last.fm tag columns (0–1 floats) are already
  on comparable scales and do not need scaling.
- The engineered features (mood_score, energy_proxy, release_recency) are
  also already on [-1,1] or [0,1] and are left unscaled.
- The fitted scaler is serialised so it can be reused at inference time
  when assigning a new track to a cluster.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from features import get_ml_feature_cols, METADATA_COLS, NEEDS_SCALING

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def get_model_input_cols(df: pd.DataFrame) -> list[str]:
    """
    Return the ordered list of columns fed to the clustering model.

    Columns in NEEDS_SCALING are replaced by their scaled versions.
    All other feature columns are passed through unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        The full feature DataFrame from features.build_feature_dataframe().

    Returns
    -------
    list[str]
    """
    all_feature_cols = get_ml_feature_cols(df)
    base = [c for c in all_feature_cols if c not in NEEDS_SCALING]
    scaled = [f"{c}_scaled" for c in NEEDS_SCALING if c in df.columns]
    return base + scaled


def fit_and_scale(
    df: pd.DataFrame,
    save_scaler: bool = True,
) -> tuple[np.ndarray, StandardScaler, list[str]]:
    """
    Fit a StandardScaler on NEEDS_SCALING columns, apply it, and return
    the scaled feature matrix ready for clustering.

    Parameters
    ----------
    df : pd.DataFrame
        Full feature DataFrame.
    save_scaler : bool
        If True, serialises the fitted scaler to models/scaler.pkl.

    Returns
    -------
    tuple:
        X : np.ndarray, shape (n_samples, n_features)
        scaler : StandardScaler (fitted)
        feature_cols : list[str] (ordered column names for X)
    """
    df = df.copy()

    # Only scale columns that are actually present in the DataFrame
    cols_to_scale = [c for c in NEEDS_SCALING if c in df.columns]

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[cols_to_scale])

    for i, col in enumerate(cols_to_scale):
        df[f"{col}_scaled"] = scaled_values[:, i]

    feature_cols = get_model_input_cols(df)

    # Verify
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns after scaling: {missing}")

    X = df[feature_cols].fillna(0).values.astype(np.float64)

    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Scaled columns: {cols_to_scale}")

    if save_scaler:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        scaler_path = MODELS_DIR / "scaler.pkl"
        joblib.dump((scaler, cols_to_scale), scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")

    return X, scaler, feature_cols


def transform_single_track(
    track_features: dict,
    feature_cols: list[str],
    scaler: StandardScaler | None = None,
    cols_to_scale: list[str] | None = None,
) -> np.ndarray:
    """
    Prepare a single track's features for cluster prediction at inference time.

    Parameters
    ----------
    track_features : dict
        Must contain all feature_cols as keys (missing keys default to 0).
    feature_cols : list[str]
        Ordered feature column names (from get_model_input_cols).
    scaler : StandardScaler, optional
        Fitted scaler. If None, loads from models/scaler.pkl.
    cols_to_scale : list[str], optional
        Which columns to scale. Loaded from disk if not provided.

    Returns
    -------
    np.ndarray, shape (1, n_features)
    """
    if scaler is None:
        scaler_path = MODELS_DIR / "scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError("No fitted scaler found. Run fit_and_scale() first.")
        scaler, cols_to_scale = joblib.load(scaler_path)

    row = {col: track_features.get(col, 0.0) for col in feature_cols}
    df_single = pd.DataFrame([row])

    # Apply scaler to the appropriate columns
    for i, col in enumerate(cols_to_scale):
        scaled_col = f"{col}_scaled"
        if scaled_col in feature_cols and col in df_single.columns:
            val = df_single[[col]].values
            df_single[scaled_col] = scaler.transform(
                np.zeros((1, len(cols_to_scale)))
            )[0][i]  # placeholder; real transform below

    # Proper transform
    raw_vals = df_single[[c for c in cols_to_scale if c in df_single.columns]].fillna(0).values
    if raw_vals.shape[1] > 0:
        scaled_vals = scaler.transform(raw_vals)[0]
        for i, col in enumerate(cols_to_scale):
            df_single[f"{col}_scaled"] = scaled_vals[i]

    X = df_single[feature_cols].fillna(0).values.astype(np.float64)
    return X


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from features import build_feature_dataframe

    df = build_feature_dataframe(save=False)
    X, scaler, cols = fit_and_scale(df)

    print(f"X shape      : {X.shape}")
    print(f"Feature count: {len(cols)}")
    print(f"Sample row   : {X[0][:10]}...")