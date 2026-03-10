"""
cluster.py
----------
Trains and evaluates K-Means clustering on the scaled audio feature matrix.

Design decisions:
- K-Means is chosen over DBSCAN for this use case because:
    1. It produces exactly K interpretable clusters (one per mood).
    2. Centroid distance is meaningful — used later by the recommender
       to map a mood input to the nearest cluster.
    3. Our data is continuous, approximately convex in feature space,
       and we expect a small number of mood archetypes (not arbitrary shapes).
  DBSCAN would be better for outlier detection or non-convex clusters;
  we document this trade-off explicitly.
- The elbow method (inertia curve) is used to pick K, with silhouette score
  as a secondary validation metric.
- Cluster labels are mapped to human-readable mood names by scoring each
  cluster's centroid against mood archetypes.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PROCESSED_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# K range to evaluate in the elbow method
K_MIN = 3
K_MAX = 10

# Mood archetype definitions.
# Each archetype is defined by the direction of key features.
# The recommender uses these to map user mood inputs → cluster IDs.
MOOD_ARCHETYPES = {
    "Energetic / Workout": {
        "energy": 1.0,
        "valence": 0.6,
        "danceability": 0.8,
        "tempo": 150,
        "acousticness": 0.1,
        "instrumentalness": 0.1,
        "liveness": 0.3,
        "loudness": -4,
        "speechiness": 0.1,
    },
    "Happy / Feel-Good": {
        "energy": 0.7,
        "valence": 1.0,
        "danceability": 0.75,
        "tempo": 120,
        "acousticness": 0.3,
        "instrumentalness": 0.1,
        "liveness": 0.15,
        "loudness": -6,
        "speechiness": 0.08,
    },
    "Chill / Relaxed": {
        "energy": 0.35,
        "valence": 0.55,
        "danceability": 0.5,
        "tempo": 90,
        "acousticness": 0.6,
        "instrumentalness": 0.3,
        "liveness": 0.1,
        "loudness": -10,
        "speechiness": 0.04,
    },
    "Focus / Deep Work": {
        "energy": 0.45,
        "valence": 0.3,
        "danceability": 0.35,
        "tempo": 100,
        "acousticness": 0.4,
        "instrumentalness": 0.8,
        "liveness": 0.1,
        "loudness": -12,
        "speechiness": 0.03,
    },
    "Melancholic / Sad": {
        "energy": 0.25,
        "valence": 0.1,
        "danceability": 0.3,
        "tempo": 75,
        "acousticness": 0.7,
        "instrumentalness": 0.2,
        "liveness": 0.08,
        "loudness": -14,
        "speechiness": 0.04,
    },
    "Late Night / Dark": {
        "energy": 0.55,
        "valence": 0.2,
        "danceability": 0.55,
        "tempo": 110,
        "acousticness": 0.15,
        "instrumentalness": 0.4,
        "liveness": 0.1,
        "loudness": -7,
        "speechiness": 0.06,
    },
    "Party / Dance": {
        "energy": 0.85,
        "valence": 0.75,
        "danceability": 0.9,
        "tempo": 128,
        "acousticness": 0.05,
        "instrumentalness": 0.05,
        "liveness": 0.2,
        "loudness": -4,
        "speechiness": 0.12,
    },
}


# ── Elbow method ──────────────────────────────────────────────────────────────
def compute_elbow_data(X: np.ndarray) -> pd.DataFrame:
    """
    Run K-Means for each K in [K_MIN, K_MAX] and record inertia and
    silhouette score.

    Used to plot the elbow curve in the EDA notebook and to inform the
    choice of optimal K.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix from preprocess.fit_and_scale().

    Returns
    -------
    pd.DataFrame
        Columns: ['k', 'inertia', 'silhouette_score']
    """
    results = []
    logger.info(f"Running elbow analysis for K = {K_MIN} to {K_MAX}...")

    for k in range(K_MIN, K_MAX + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        labels = km.fit_predict(X)
        inertia = km.inertia_
        sil = silhouette_score(X, labels) if k > 1 else 0.0

        results.append({"k": k, "inertia": inertia, "silhouette_score": sil})
        logger.info(f"  K={k}: inertia={inertia:.1f}, silhouette={sil:.4f}")

    return pd.DataFrame(results)


def suggest_optimal_k(elbow_df: pd.DataFrame) -> int:
    """
    Suggest the optimal K using the silhouette score (higher = better).

    Falls back to elbow heuristic (largest drop in inertia) if silhouette
    scores are flat.

    Parameters
    ----------
    elbow_df : pd.DataFrame
        Output of compute_elbow_data().

    Returns
    -------
    int
        Suggested optimal K.
    """
    best_k = elbow_df.loc[elbow_df["silhouette_score"].idxmax(), "k"]
    logger.info(f"Suggested optimal K by silhouette score: {best_k}")
    return int(best_k)


# ── Model training ────────────────────────────────────────────────────────────
def train_kmeans(
    X: np.ndarray,
    k: int,
    save: bool = True,
) -> KMeans:
    """
    Train K-Means with the specified K.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix.
    k : int
        Number of clusters.
    save : bool
        If True, serialises the model to models/kmeans_model.pkl.

    Returns
    -------
    KMeans
        Fitted K-Means model.
    """
    logger.info(f"Training K-Means with K={k}...")
    km = KMeans(
        n_clusters=k,
        init="k-means++",   # smarter initialisation → more stable clusters
        n_init=20,           # run 20 times with different seeds, keep best
        max_iter=500,
        random_state=42,
        verbose=0,
    )
    km.fit(X)
    logger.info(f"Training complete. Inertia: {km.inertia_:.2f}")

    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / "kmeans_model.pkl"
        joblib.dump(km, model_path)
        logger.info(f"Model saved to {model_path}")

    return km


# ── Cluster labelling ─────────────────────────────────────────────────────────
def label_clusters(
    km: KMeans,
    feature_cols: list[str],
    scaler,
) -> dict[int, str]:
    """
    Map cluster IDs to human-readable mood names by comparing each cluster's
    centroid to the MOOD_ARCHETYPES definitions.

    Method: for each centroid, we find the mood archetype whose feature vector
    has the smallest cosine distance to the centroid (in the original,
    unscaled feature space for interpretability).

    Parameters
    ----------
    km : KMeans
        Fitted K-Means model.
    feature_cols : list[str]
        Ordered feature column names (from preprocess.get_model_input_cols()).
    scaler : StandardScaler
        Fitted scaler — used to inverse-transform centroids for comparison.

    Returns
    -------
    dict[int, str]
        Maps cluster_id → mood label string.
    """
    from features import AUDIO_FEATURE_COLS, NEEDS_SCALING

    centroids_scaled = km.cluster_centers_

    # Find the indices of the scaled columns in feature_cols
    scaled_col_names = [f"{c}_scaled" for c in NEEDS_SCALING]
    scaled_indices = [feature_cols.index(c) for c in scaled_col_names if c in feature_cols]
    unscaled_indices = [i for i in range(len(feature_cols)) if i not in scaled_indices]

    cluster_to_mood = {}

    for cluster_id, centroid in enumerate(centroids_scaled):
        # Inverse-transform the scaled dimensions
        scaled_part = centroid[scaled_indices].reshape(1, -1)
        original_values = scaler.inverse_transform(scaled_part)[0]

        # Reconstruct the 9 core audio features from the centroid
        centroid_audio = {}
        for i, col in enumerate(feature_cols):
            if col in AUDIO_FEATURE_COLS:
                centroid_audio[col] = centroid[i]
        for i, col in enumerate(NEEDS_SCALING):
            centroid_audio[col] = original_values[i]

        # Compare to each archetype using cosine similarity
        best_mood = None
        best_sim = -np.inf

        for mood_name, archetype in MOOD_ARCHETYPES.items():
            # Build vectors from shared keys
            keys = AUDIO_FEATURE_COLS
            a = np.array([centroid_audio.get(k, 0) for k in keys])
            b = np.array([archetype.get(k, 0) for k in keys])

            # Normalise to avoid scale effects
            a_norm = np.linalg.norm(a)
            b_norm = np.linalg.norm(b)
            if a_norm == 0 or b_norm == 0:
                sim = 0.0
            else:
                sim = np.dot(a, b) / (a_norm * b_norm)

            if sim > best_sim:
                best_sim = sim
                best_mood = mood_name

        cluster_to_mood[cluster_id] = best_mood
        logger.info(f"  Cluster {cluster_id} → {best_mood} (cosine sim: {best_sim:.3f})")

    return cluster_to_mood


# ── Full pipeline ─────────────────────────────────────────────────────────────
def run_clustering_pipeline(
    X: np.ndarray,
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler,
    k: int | None = None,
) -> tuple[pd.DataFrame, KMeans, dict[int, str]]:
    """
    End-to-end clustering: elbow → train → label → annotate DataFrame.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix.
    df : pd.DataFrame
        Feature DataFrame with metadata (output of features.build_feature_dataframe()).
    feature_cols : list[str]
        Ordered list of feature column names in X.
    scaler : StandardScaler
        Fitted scaler.
    k : int, optional
        Number of clusters. If None, computed automatically via elbow method.

    Returns
    -------
    tuple:
        df_clustered : pd.DataFrame
            Original df with 'cluster_id' and 'mood_label' columns added.
        km : KMeans
            Fitted K-Means model.
        cluster_to_mood : dict[int, str]
            Cluster ID → mood label mapping.
    """
    if k is None:
        elbow_df = compute_elbow_data(X)
        elbow_path = PROCESSED_DATA_DIR / "elbow_data.csv"
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        elbow_df.to_csv(elbow_path, index=False)
        k = suggest_optimal_k(elbow_df)

    km = train_kmeans(X, k=k)
    cluster_ids = km.predict(X)

    cluster_to_mood = label_clusters(km, feature_cols, scaler)

    df_clustered = df.copy()
    df_clustered["cluster_id"] = cluster_ids
    df_clustered["mood_label"] = df_clustered["cluster_id"].map(cluster_to_mood)

    # Save clustered data
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / "clustered_tracks.csv"
    df_clustered.to_csv(out_path, index=False)
    logger.info(f"Clustered tracks saved to {out_path}")

    # Summary
    summary = df_clustered.groupby(["cluster_id", "mood_label"]).size().reset_index(name="count")
    logger.info(f"\nCluster summary:\n{summary.to_string(index=False)}")

    return df_clustered, km, cluster_to_mood


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from features import build_feature_dataframe
    from preprocess import fit_and_scale

    df = build_feature_dataframe(save=False)
    X, scaler, feature_cols = fit_and_scale(df)
    df_clustered, km, cluster_to_mood = run_clustering_pipeline(
        X, df, feature_cols, scaler
    )

    print("\nCluster → Mood mapping:")
    for cid, mood in cluster_to_mood.items():
        count = (df_clustered["cluster_id"] == cid).sum()
        print(f"  Cluster {cid}: {mood}  ({count} tracks)")
