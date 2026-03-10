# MoodSync - Spotify + Last.fm ML Playlist Generator

MoodSync is an end-to-end machine learning project that analyzes your Spotify library,
clusters tracks into mood groups, and generates playlists from natural-language mood input.

## What It Does

1. Fetches your liked songs and top tracks from Spotify.
2. Fetches Last.fm tags for those tracks.
3. Builds a feature table from:
   - Spotify metadata (for example: popularity, duration, release year)
   - Last.fm tag vectors (`tag_*` mood/activity features)
   - Engineered features (`mood_score`, `energy_proxy`, `release_recency`)
4. Scales selected numeric columns and trains K-Means.
5. Labels clusters with mood archetypes.
6. Visualizes clusters with PCA and UMAP.
7. Recommends tracks for a mood query.
8. Exports recommendations to Spotify playlists.

## ML Techniques Used

| Technique | Purpose | Why this choice |
|---|---|---|
| K-Means | Mood clustering | Interpretable clusters and centroid-based matching |
| StandardScaler | Feature normalization | Applied to `popularity`, `duration_ms`, `listeners_log`, `playcount_log` |
| PCA | Dimensionality reduction | Fast, linear, and easy to interpret |
| UMAP | Cluster visualization | Better local structure separation for interactive exploration |
| Cosine similarity | Mood-to-cluster matching | Works well for direction-based vector similarity |
| Elbow + silhouette | K selection | Practical dual check for cluster quality |

## Project Structure

```text
moodsync/
|-- src/
|   |-- spotify_client.py   # Spotify auth, track fetch, playlist export
|   |-- lastfm_client.py    # Last.fm tag fetch + normalization
|   |-- features.py         # Feature building + engineering
|   |-- preprocess.py       # Scaling + model input prep
|   |-- cluster.py          # K-Means training + cluster labeling
|   |-- recommender.py      # Mood query -> cluster -> playlist
|   `-- visualize.py        # PCA/UMAP/radar/elbow plots
|-- app/
|   `-- streamlit_app.py    # Interactive Streamlit dashboard
|-- data/
|   |-- raw/                # Raw API outputs (gitignored)
|   `-- processed/          # Derived datasets + plot artifacts
|-- models/                 # Serialized model + scaler (gitignored)
|-- notebooks/
|-- requirements.txt
|-- env.example
`-- README.md
```

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows (PowerShell):

```bash
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure credentials

Copy `env.example` to `.env`, then fill in:

- `SPOTIPY_CLIENT_ID`
- `SPOTIPY_CLIENT_SECRET`
- `SPOTIPY_REDIRECT_URI`
- `LASTFM_API_KEY`
- `LASTFM_API_SECRET`

### 4. Run the pipeline

```bash
# 1) Pull Spotify tracks
python src/spotify_client.py

# 2) Pull Last.fm tags for those tracks
python src/lastfm_client.py

# 3) Train clustering model and write clustered outputs
python src/cluster.py

# 4) Launch UI
streamlit run app/streamlit_app.py
```

## Core Outputs

- `data/raw/tracks.json`
- `data/raw/lastfm_tags.json`
- `data/processed/features.csv`
- `data/processed/elbow_data.csv`
- `data/processed/clustered_tracks.csv`
- `models/scaler.pkl`
- `models/kmeans_model.pkl`

## Notes

- Spotify playlist export requires valid playlist scopes and an approved user account
  in your Spotify Developer app if the app is in development mode.
- If OAuth scopes change, delete the local Spotipy token cache and re-authenticate.

## License

MIT
