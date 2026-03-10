"""
visualize.py
------------
Generates interactive visualisations of the cluster structure using
PCA and UMAP dimensionality reduction.

Design decisions:
- PCA is run first: fast, linear, fully explainable. Good for a quick
  sanity check and for explaining variance to non-technical audiences.
- UMAP is run second: non-linear, better at preserving local cluster
  structure. Produces more visually separated clusters at the cost of
  being harder to explain mathematically.
- All plots use Plotly for interactivity (hover = track name + artist),
  which is critical for the Streamlit demo.
- Plots are saved as standalone HTML so they can be included in a portfolio
  without running the full app.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


# â”€â”€ PCA â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def run_pca(X: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, PCA]:
    """
    Fit PCA on the scaled feature matrix and return 2D projections.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix from preprocess.fit_and_scale().
    n_components : int
        Number of principal components to keep.

    Returns
    -------
    tuple[np.ndarray, PCA]
        (X_pca, fitted_pca_object)
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    variance_explained = pca.explained_variance_ratio_.sum() * 100
    logger.info(
        f"PCA: {n_components} components explain "
        f"{variance_explained:.1f}% of variance."
    )
    return X_pca, pca


def _build_hover_data(plot_df: pd.DataFrame, preferred: list[str], hidden_axes: list[str]) -> dict:
    """Build Plotly hover_data dict using only columns that exist."""
    hover_data = {col: ":.2f" for col in preferred if col in plot_df.columns}
    for axis in hidden_axes:
        if axis in plot_df.columns:
            hover_data[axis] = False
    return hover_data


def run_umap(X: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    """
    Run UMAP dimensionality reduction.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix.
    n_neighbors : int
        Controls how UMAP balances local vs global structure.
        Lower = more local. Default 15 works well for music data.
    min_dist : float
        Minimum distance between points in the embedding.
        Lower = tighter clusters.

    Returns
    -------
    np.ndarray
        2D UMAP embedding, shape (n_samples, 2).
    """
    try:
        import umap
    except ImportError:
        raise ImportError("umap-learn is required. Run: pip install umap-learn")

    logger.info("Running UMAP (this may take 30â€“60 seconds for large datasets)...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
        metric="cosine",    # cosine distance suits normalised audio features
    )
    X_umap = reducer.fit_transform(X)
    logger.info("UMAP complete.")
    return X_umap


# â”€â”€ Plot builders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def plot_pca_clusters(
    df: pd.DataFrame,
    X_pca: np.ndarray,
    pca: PCA,
    save: bool = True,
) -> go.Figure:
    """
    Create an interactive PCA scatter plot coloured by mood label.

    Parameters
    ----------
    df : pd.DataFrame
        Clustered tracks DataFrame (must have 'mood_label', 'name', 'artist').
    X_pca : np.ndarray
        2D PCA projections.
    pca : PCA
        Fitted PCA object (used for axis labels).
    save : bool
        If True, saves HTML to data/processed/pca_plot.html.

    Returns
    -------
    go.Figure
        Plotly figure.
    """
    plot_df = df.copy()
    plot_df["PC1"] = X_pca[:, 0]
    plot_df["PC2"] = X_pca[:, 1]
    plot_df["hover"] = plot_df["name"] + " â€” " + plot_df["artist"]

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100

    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="mood_label",
        hover_name="hover",
        hover_data=_build_hover_data(plot_df, ["energy", "valence", "mood_score", "energy_proxy"], ["PC1", "PC2"]),
        title="PCA â€” Music Mood Clusters",
        labels={
            "PC1": f"PC1 ({var1:.1f}% variance)",
            "PC2": f"PC2 ({var2:.1f}% variance)",
            "mood_label": "Mood",
        },
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    fig.update_layout(
        font_family="monospace",
        title_font_size=18,
        legend_title_text="Mood Cluster",
    )

    if save:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(OUTPUTS_DIR / "pca_plot.html"))
        logger.info("PCA plot saved.")

    return fig


def plot_umap_clusters(
    df: pd.DataFrame,
    X_umap: np.ndarray,
    save: bool = True,
) -> go.Figure:
    """
    Create an interactive UMAP scatter plot coloured by mood label.

    Parameters
    ----------
    df : pd.DataFrame
        Clustered tracks DataFrame.
    X_umap : np.ndarray
        2D UMAP projections.
    save : bool
        If True, saves HTML to data/processed/umap_plot.html.

    Returns
    -------
    go.Figure
        Plotly figure.
    """
    plot_df = df.copy()
    plot_df["UMAP1"] = X_umap[:, 0]
    plot_df["UMAP2"] = X_umap[:, 1]
    plot_df["hover"] = plot_df["name"] + " â€” " + plot_df["artist"]

    fig = px.scatter(
        plot_df,
        x="UMAP1",
        y="UMAP2",
        color="mood_label",
        hover_name="hover",
        hover_data=_build_hover_data(
            plot_df,
            ["energy", "valence", "danceability", "mood_score", "energy_proxy", "release_recency"],
            ["UMAP1", "UMAP2"],
        ),
        title="UMAP â€” Music Mood Clusters",
        labels={"mood_label": "Mood"},
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_traces(marker=dict(size=5, opacity=0.75))
    fig.update_layout(
        font_family="monospace",
        title_font_size=18,
        legend_title_text="Mood Cluster",
    )

    if save:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(OUTPUTS_DIR / "umap_plot.html"))
        logger.info("UMAP plot saved.")

    return fig


def plot_elbow_curve(elbow_df: pd.DataFrame, save: bool = True) -> go.Figure:
    """
    Plot the K-Means elbow curve (inertia + silhouette score vs K).

    Parameters
    ----------
    elbow_df : pd.DataFrame
        Output of cluster.compute_elbow_data() with columns k, inertia,
        silhouette_score.
    save : bool
        If True, saves HTML to data/processed/elbow_plot.html.

    Returns
    -------
    go.Figure
        Plotly figure with dual y-axis.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=elbow_df["k"],
        y=elbow_df["inertia"],
        name="Inertia",
        mode="lines+markers",
        marker=dict(size=8),
        line=dict(color="#FF6B6B"),
        yaxis="y1",
    ))

    fig.add_trace(go.Scatter(
        x=elbow_df["k"],
        y=elbow_df["silhouette_score"],
        name="Silhouette Score",
        mode="lines+markers",
        marker=dict(size=8),
        line=dict(color="#4ECDC4", dash="dash"),
        yaxis="y2",
    ))

    fig.update_layout(
        title="K-Means Elbow Curve",
        xaxis=dict(title="Number of Clusters (K)", tickmode="linear"),
        yaxis=dict(title="Inertia", titlefont=dict(color="#FF6B6B")),
        yaxis2=dict(
            title="Silhouette Score",
            titlefont=dict(color="#4ECDC4"),
            overlaying="y",
            side="right",
        ),
        template="plotly_dark",
        font_family="monospace",
        legend=dict(x=0.7, y=0.95),
    )

    if save:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(OUTPUTS_DIR / "elbow_plot.html"))
        logger.info("Elbow plot saved.")

    return fig


def plot_feature_radar(
    cluster_to_mood: dict,
    km,
    feature_cols: list[str],
    scaler,
    save: bool = True,
) -> go.Figure:
    """
    Radar chart showing each cluster's feature profile.

    Improvements over the original version:
    - Uses unique legend labels (cluster id + mood).
    - Selects features by between-cluster variance, so the chart highlights
      dimensions that actually separate clusters.
    - Min-max normalises each selected feature across clusters so all axes
      are comparable on [0, 1].
    """
    del scaler  # kept in signature for compatibility with existing callers

    # Prefer intuitive, human-readable dimensions when available.
    preferred_features = [
        "mood_score", "energy_proxy", "release_recency",
        "tag_energetic", "tag_happy", "tag_chill", "tag_sad",
        "tag_focus", "tag_party", "tag_dark",
    ]

    candidate_features = [
        c for c in preferred_features
        if c in feature_cols and not c.endswith("_scaled")
    ]

    if not candidate_features:
        candidate_features = [
            c for c in feature_cols
            if not c.endswith("_scaled") and c not in {"listeners_log", "playcount_log"}
        ]

    if not candidate_features:
        raise ValueError("No compatible features available for radar plot.")

    ordered_cluster_ids = sorted(cluster_to_mood.keys())
    centroids = km.cluster_centers_

    # Build [n_clusters, n_features] matrix from centroids in model feature space.
    feature_idx = [feature_cols.index(f) for f in candidate_features]
    raw_matrix = np.array([
        [centroids[cid][idx] for idx in feature_idx]
        for cid in ordered_cluster_ids
    ], dtype=float)

    # Pick the most discriminative axes (top variance across clusters).
    variances = raw_matrix.var(axis=0)
    top_n = min(8, len(candidate_features))
    top_idx = np.argsort(variances)[::-1][:top_n]

    radar_features = [candidate_features[i] for i in top_idx]
    selected = raw_matrix[:, top_idx]

    # Normalise each axis to [0, 1] across clusters for visual comparability.
    mins = selected.min(axis=0)
    maxs = selected.max(axis=0)
    denom = np.where((maxs - mins) == 0, 1.0, (maxs - mins))
    norm = (selected - mins) / denom

    fig = go.Figure()
    for i, cluster_id in enumerate(ordered_cluster_ids):
        mood_label = cluster_to_mood.get(cluster_id, "Unknown")
        values = norm[i].tolist()
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=radar_features + [radar_features[0]],
            name=f"Cluster {cluster_id}: {mood_label}",
            fill="toself",
            opacity=0.6,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Cluster Audio Feature Profiles (Normalised)",
        template="plotly_dark",
        font_family="monospace",
    )

    if save:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(OUTPUTS_DIR / "radar_plot.html"))
        logger.info("Radar plot saved.")

    return fig

# â”€â”€ Entry point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from features import build_feature_dataframe
    from preprocess import fit_and_scale
    from cluster import run_clustering_pipeline

    df = build_feature_dataframe(save=False)
    X, scaler, feature_cols = fit_and_scale(df)
    df_clustered, km, cluster_to_mood = run_clustering_pipeline(X, df, feature_cols, scaler)

    X_pca, pca = run_pca(X)
    X_umap = run_umap(X)

    plot_pca_clusters(df_clustered, X_pca, pca)
    plot_umap_clusters(df_clustered, X_umap)
    plot_feature_radar(cluster_to_mood, km, feature_cols, scaler)

    print("All visualisations saved to data/processed/")

