"""
streamlit_app.py
----------------
MoodSync â€” Interactive Spotify Playlist Generator

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st
import pandas as pd
import plotly.express as px

from recommender import get_playlist, get_available_moods, load_model_artifacts, load_clustered_tracks
from preprocess import get_model_input_cols
from visualize import run_pca, run_umap, plot_pca_clusters, plot_umap_clusters, plot_feature_radar
from cluster import MOOD_ARCHETYPES
from features import build_feature_dataframe
from preprocess import fit_and_scale
from cluster import run_clustering_pipeline

# â”€â”€ Page config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.set_page_config(
    page_title="MoodSync",
    page_icon="\U0001F3B5",
    layout="wide",
    initial_sidebar_state="expanded",
)

# â”€â”€ Styling â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'Space Mono', monospace;
}
.track-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 14px;
}
.track-card img {
    width: 52px;
    height: 52px;
    border-radius: 6px;
    object-fit: cover;
}
.track-name {
    font-weight: 600;
    font-size: 15px;
    color: #fff;
    margin: 0;
}
.track-artist {
    font-size: 13px;
    color: rgba(255,255,255,0.55);
    margin: 0;
}
.mood-badge {
    display: inline-block;
    background: rgba(100,220,150,0.15);
    color: #64DC96;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 12px;
    font-family: 'Space Mono', monospace;
    border: 1px solid rgba(100,220,150,0.3);
}
.stat-box {
    background: rgba(255,255,255,0.04);
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.07);
}
.stat-number {
    font-size: 28px;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
    color: #64DC96;
}
.stat-label {
    font-size: 12px;
    color: rgba(255,255,255,0.5);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)


# â”€â”€ Session state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@st.cache_resource(show_spinner="Loading model artifacts...")
def load_artifacts():
    km, scaler, _ = load_model_artifacts()
    df_clustered = load_clustered_tracks()
    return km, scaler, df_clustered


@st.cache_resource(show_spinner="Computing UMAP (one-time, ~30s)...")
def get_umap_data(X):
    return run_umap(X)


# â”€â”€ Sidebar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with st.sidebar:
    st.markdown("# \U0001F3B5 MoodSync")
    st.markdown("*Spotify ML Playlist Generator*")
    st.divider()

    st.markdown("### How it works")
    st.markdown("""
1. Your liked songs are pulled from Spotify
2. Audio features are extracted (energy, valence, tempo...)
3. K-Means clusters them into mood groups
4. You pick a mood -> get your playlist
    """)

    st.divider()

    st.markdown("### Re-run pipeline")
    if st.button("\U0001F504 Refresh data from Spotify", use_container_width=True):
        st.info("Run `python src/spotify_client.py` then restart the app.")

    st.divider()
    st.caption("Built with Spotipy | scikit-learn | UMAP | Streamlit")


# â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.title("\U0001F3B5 MoodSync")
st.markdown("##### ML-powered playlists from your Spotify library")
st.divider()

# Check if model is trained
try:
    km, scaler, df_clustered = load_artifacts()
    model_ready = True
except FileNotFoundError as e:
    model_ready = False
    st.error(f"""
    **Model not trained yet.**

    Run the pipeline first:
    ```bash
    python src/spotify_client.py    # fetch your data
    python src/cluster.py           # train the model
    ```
    Then refresh this page.
    """)

if model_ready:

    # â”€â”€ Stats row â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    n_tracks = len(df_clustered)
    n_clusters = df_clustered["cluster_id"].nunique()
    mood_labels = df_clustered["mood_label"].unique().tolist()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{n_tracks}</div>
            <div class="stat-label">Tracks Analysed</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{n_clusters}</div>
            <div class="stat-label">Mood Clusters</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        if "energy_proxy" in df_clustered.columns:
            avg_energy = df_clustered["energy_proxy"].mean()
        elif "tag_energetic" in df_clustered.columns:
            avg_energy = df_clustered["tag_energetic"].mean()
        else:
            avg_energy = 0
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{avg_energy:.2f}</div>
            <div class="stat-label">Avg Energy</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["\U0001F3A7 Generate Playlist", "\U0001F4CA Cluster Visualisation", "\U0001F50D Explore Library"])
    # â”€â”€ Tab 1: Playlist generator â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with tab1:
        st.markdown("### What's your mood?")

        col_input, col_settings = st.columns([2, 1])

        with col_input:
            # Quick mood buttons
            st.markdown("**Quick select:**")
            mood_options = list(MOOD_ARCHETYPES.keys())
            selected_mood = st.selectbox(
                "Choose a mood",
                options=mood_options,
                index=2,  # default: Chill / Relaxed
                label_visibility="collapsed"
            )

            # Or free text
            custom_mood = st.text_input(
                "Or type anything:",
                placeholder="e.g. 'late night drive', 'studying', 'gym session'...",
            )

            mood_query = custom_mood if custom_mood.strip() else selected_mood

        with col_settings:
            n_tracks = st.slider("Tracks", min_value=5, max_value=50, value=20, step=5)
            diversity = st.slider(
                "Diversity",
                min_value=0.0, max_value=1.0, value=0.2, step=0.1,
                help="Higher = more varied picks from the cluster"
            )

        generate_btn = st.button("\u2728 Generate Playlist", type="primary", use_container_width=True)

        if generate_btn or "last_playlist" in st.session_state:

            if generate_btn:
                with st.spinner(f"Building your {mood_query} playlist..."):
                    playlist = get_playlist(
                        mood_query,
                        n_tracks=n_tracks,
                        diversity=diversity,
                        km=km,
                        scaler=scaler,
                        df_clustered=df_clustered,
                    )
                st.session_state["last_playlist"] = playlist
                st.session_state["last_mood"] = mood_query
            else:
                playlist = st.session_state["last_playlist"]
                mood_query = st.session_state.get("last_mood", mood_query)

            if not playlist.empty:
                detected_mood = playlist["mood_label"].iloc[0] if "mood_label" in playlist.columns else mood_query
                st.markdown(f"**{len(playlist)} tracks** for: <span class='mood-badge'>{detected_mood}</span>", unsafe_allow_html=True)
                st.markdown("")

                # Export to Spotify button
                if "uri" in playlist.columns:
                    track_ids = [uri.split(":")[-1] for uri in playlist["uri"].dropna()]
                    if track_ids:
                        col_export, col_dl = st.columns([1, 1])
                        with col_export:
                            if st.button("\u2795 Export to Spotify", use_container_width=True):
                                try:
                                    from spotify_client import get_spotify_client, create_spotify_playlist
                                    client = get_spotify_client()
                                    url = create_spotify_playlist(
                                        client,
                                        track_ids,
                                        name=f"MoodSync â€” {detected_mood}",
                                        description=f"Generated by MoodSync ML for mood: {mood_query}",
                                    )
                                    st.success(f"Playlist created! [Open in Spotify]({url})")
                                except Exception as e:
                                    st.error(f"Export failed: {e}")
                        with col_dl:
                            csv = playlist[["name", "artist", "album"]].to_csv(index=False)
                            st.download_button(
                                "Download CSV",
                                data=csv,
                                file_name=f"moodsync_{mood_query.replace(' ','_')}.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )

                st.markdown("")

                # Track list
                for _, row in playlist.iterrows():
                    art_url = row.get("album_art_url", "")
                    img_tag = f'<img src="{art_url}">' if art_url and str(art_url) != "nan" else \
                              '<div style="width:52px;height:52px;background:#333;border-radius:6px;"></div>'
                    st.markdown(f"""
                    <div class="track-card">
                        {img_tag}
                        <div>
                            <p class="track-name">{row.get("name", "")}</p>
                            <p class="track-artist">{row.get("artist", "")} \u00b7 {row.get("album", "")}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Feature distribution of playlist
                st.markdown("")
                st.markdown("#### Playlist audio profile")
                feature_cols_viz = ["energy", "valence", "danceability", "acousticness", "instrumentalness"]
                feature_cols_viz = [c for c in feature_cols_viz if c in playlist.columns]
                if feature_cols_viz:
                    means = playlist[feature_cols_viz].mean().reset_index()
                    means.columns = ["Feature", "Value"]
                    fig = px.bar(
                        means, x="Feature", y="Value",
                        color="Value",
                        color_continuous_scale="Teal",
                        range_y=[0, 1],
                        template="plotly_dark",
                        title="",
                    )
                    fig.update_layout(
                        showlegend=False,
                        coloraxis_showscale=False,
                        height=250,
                        margin=dict(t=10, b=10, l=10, r=10),
                        font_family="monospace",
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No tracks found for this mood. Try a different one.")

    # â”€â”€ Tab 2: Cluster visualisation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with tab2:
        st.markdown("### Cluster Visualisation")
        st.markdown(
            "Each point is a track from your library. Colours = mood clusters discovered by K-Means. "
            "Hover to see track details."
        )

        viz_type = st.radio(
            "Projection method:",
            ["UMAP (recommended)", "PCA"],
            horizontal=True,
        )

        feature_cols = get_model_input_cols(df_clustered.drop(columns=["cluster_id", "mood_label"], errors="ignore"))
        available_cols = [c for c in feature_cols if c in df_clustered.columns]

        if available_cols:
            X_viz = df_clustered[available_cols].fillna(0).values

            if "UMAP" in viz_type:
                with st.spinner("Computing UMAP..."):
                    X_umap = get_umap_data(X_viz)
                fig = plot_umap_clusters(df_clustered, X_umap, save=False)
            else:
                X_pca, pca = run_pca(X_viz)
                fig = plot_pca_clusters(df_clustered, X_pca, pca, save=False)

            st.plotly_chart(fig, use_container_width=True)

            # Radar chart
            st.markdown("### Cluster Audio Profiles")
            st.markdown("Radar chart showing what makes each mood cluster distinct.")

            cluster_to_mood = dict(zip(
                df_clustered["cluster_id"].unique(),
                df_clustered.groupby("cluster_id")["mood_label"].first()
            ))
            fig_radar = plot_feature_radar(cluster_to_mood, km, feature_cols, scaler, save=False)
            st.plotly_chart(fig_radar, use_container_width=True)

        else:
            st.warning("Feature columns not found in clustered data.")

    # â”€â”€ Tab 3: Explore library â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with tab3:
        st.markdown("### Your Library")

        # Filter by mood
        all_moods = ["All"] + sorted(df_clustered["mood_label"].dropna().unique().tolist())
        selected_filter = st.selectbox("Filter by mood cluster:", all_moods)

        filtered = df_clustered if selected_filter == "All" else \
                   df_clustered[df_clustered["mood_label"] == selected_filter]

        st.markdown(f"**{len(filtered)} tracks**")

        display_cols = ["name", "artist", "album", "mood_label", "energy", "valence", "danceability", "tempo"]
        display_cols = [c for c in display_cols if c in filtered.columns]

        st.dataframe(
            filtered[display_cols].reset_index(drop=True),
            use_container_width=True,
            height=500,
        )

