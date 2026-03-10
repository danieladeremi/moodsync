"""
preprocess.py
-------------
Scales and prepares the feature DataFrame for clustering.

Design decisions:
- We apply StandardScaler ONLY to tempo and loudness, because they are on
  different numeric scales (BPM ~60–200, loudness ~-60–0 dB). All other
  features are already on [0, 1] and scaling them would distort their natural
  interpretability without improving cluster quality.
- We serialise the fitted scaler alongside the model so it can be reused
  at inference time (when a new track needs to be assigned to a cluster,
  it must be scaled with the same parameters as training data).
- The function returns both the scaled array (for the model) and the original
  DataFrame (for visualisation / metadata lookups).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from features import AUDIO_FEATURE_COLS, NEEDS_SCALING, get_ml_feature_cols

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# These are the feature columns we'll pass to the clustering model.
# We exclude raw tempo and loudness (they get replaced by their scaled versions),
# but keep all other originals plus the engineered features.
def get_model_input_cols() -> list[str]:
    """
    Returns the ordered list of columns fed to the clustering model.

    Raw tempo and loudness are excluded — they are replaced by
    tempo_scaled and loudness_scaled produced by fit_and_scale().
    """
    base = [c for c in get_ml_feature_cols() if c not in NEEDS_SCALING]
    scaled = [f"{c}_scaled" for c in NEEDS_SCALING]
    return base + scaled


def fit_and_scale(
    df: pd.DataFrame,
    save_scaler: bool = True,
) -> tuple[np.ndarray, StandardScaler, list[str]]:
    """
    Fit a StandardScaler on tempo and loudness, apply it, and return the
    scaled feature matrix ready for clustering.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain all columns from get_ml_feature_cols().
    save_scaler : bool
        If True, serialises the fitted scaler to models/scaler.pkl.

    Returns
    -------
    tuple:
        X : np.ndarray, shape (n_samples, n_features)
            Scaled feature matrix.
        scaler : StandardScaler
            Fitted scaler (needed to transform new tracks at inference).
        feature_cols : list[str]
            Ordered column names corresponding to X's columns.
    """
    df = df.copy()

    # Fit and apply StandardScaler to the features that need it
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[NEEDS_SCALING])

    for i, col in enumerate(NEEDS_SCALING):
        df[f"{col}_scaled"] = scaled_values[:, i]

    feature_cols = get_model_input_cols()

    # Verify all expected columns are present
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns after scaling: {missing}")

    X = df[feature_cols].values.astype(np.float64)

    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Features used: {feature_cols}")

    if save_scaler:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        scaler_path = MODELS_DIR / "scaler.pkl"
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")

    return X, scaler, feature_cols


def transform_single_track(
    track_features: dict,
    scaler: StandardScaler | None = None,
) -> np.ndarray:
    """
    Prepare a single track's features for cluster prediction at inference time.

    Parameters
    ----------
    track_features : dict
        Must contain all AUDIO_FEATURE_COLS as keys. Can be a raw Spotify
        audio features dict.
    scaler : StandardScaler, optional
        Fitted scaler. If None, loads from models/scaler.pkl.

    Returns
    -------
    np.ndarray, shape (1, n_features)
        Scaled feature vector ready for model.predict().
    """
    if scaler is None:
        scaler_path = MODELS_DIR / "scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(
                "No fitted scaler found. Run fit_and_scale() first."
            )
        scaler = joblib.load(scaler_path)

    # Build a single-row DataFrame so we can reuse feature engineering logic
    row = {col: track_features.get(col, np.nan) for col in AUDIO_FEATURE_COLS}
    df_single = pd.DataFrame([row])

    # Replicate the same engineered features as in features.py
    df_single["energy_valence"] = df_single["energy"] * df_single["valence"]
    df_single["danceability_energy"] = df_single["danceability"] * df_single["energy"]

    # tempo_norm: we don't have the training dataset's min/max here, so we use
    # a typical BPM range of 60–200 as a reasonable normalisation
    df_single["tempo_norm"] = (df_single["tempo"] - 60) / (200 - 60)
    df_single["tempo_norm"] = df_single["tempo_norm"].clip(0, 1)

    # Apply the saved StandardScaler to tempo and loudness
    scaled_values = scaler.transform(df_single[NEEDS_SCALING])
    for i, col in enumerate(NEEDS_SCALING):
        df_single[f"{col}_scaled"] = scaled_values[:, i]

    feature_cols = get_model_input_cols()
    X = df_single[feature_cols].values.astype(np.float64)
    return X


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd
    from features import build_feature_dataframe

    df = build_feature_dataframe(save=False)
    X, scaler, cols = fit_and_scale(df)

    print(f"X shape     : {X.shape}")
    print(f"Feature cols: {cols}")
    print(f"X sample row: {X[0]}")
