# 🎵 MoodSync — Spotify ML Playlist Generator

An end-to-end machine learning system that analyses your Spotify listening history,
discovers mood clusters using unsupervised learning, and generates personalised
playlists from natural language mood input.

---

## What it does

1. **Fetches** your liked songs and top tracks from Spotify via the Web API
2. **Extracts** 9 audio features per track (energy, valence, danceability, tempo, etc.)
3. **Engineers** composite features to improve cluster separation
4. **Clusters** tracks into mood groups using K-Means (optimal K via elbow method)
5. **Visualises** clusters in 2D using PCA and UMAP
6. **Recommends** a playlist given any natural language mood input
7. **Exports** the playlist directly back to your Spotify account

---

## ML Techniques Used

| Technique | Purpose | Why this choice |
|---|---|---|
| K-Means | Mood clustering | Interpretable, centroid distance meaningful for recommendation |
| StandardScaler | Feature normalisation | Applied only to tempo/loudness (different scales) |
| PCA | Dimensionality reduction | Fast, linear, explains variance quantitatively |
| UMAP | Cluster visualisation | Preserves local structure better than t-SNE for recommendation context |
| Cosine Similarity | Mood-to-cluster matching | Scale-invariant, suited to normalised feature vectors |
| Elbow + Silhouette | Optimal K selection | Dual validation avoids over/under-segmentation |

---

## Project Structure

```
moodsync/
├── src/
│   ├── spotify_client.py   # API auth, data fetching, playlist creation
│   ├── features.py         # Feature extraction and engineering
│   ├── preprocess.py       # Scaling and model input preparation
│   ├── cluster.py          # K-Means training, elbow method, mood labelling
│   ├── recommender.py      # Mood → cluster → playlist logic
│   └── visualize.py        # PCA, UMAP, radar, elbow plots (Plotly)
├── app/
│   └── streamlit_app.py    # Interactive demo dashboard
├── notebooks/
│   ├── 01_eda.ipynb        # Exploratory data analysis
│   ├── 02_clustering.ipynb # Clustering experiments
│   └── 03_umap_viz.ipynb   # Dimensionality reduction visuals
├── data/
│   ├── raw/                # Spotify API output (gitignored)
│   └── processed/          # Scaled features, cluster assignments
└── models/                 # Serialised model + scaler (gitignored)
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/yourusername/moodsync.git
cd moodsync
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Spotify credentials

```bash
cp .env.example .env
```

Edit `.env` with your credentials from [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard):

```
SPOTIPY_CLIENT_ID=your_client_id
SPOTIPY_CLIENT_SECRET=your_client_secret
SPOTIPY_REDIRECT_URI=http://127.0.0.1:8888/callback
```

### 3. Run the pipeline

```bash
# Step 1: Fetch your Spotify data (opens browser for OAuth on first run)
python src/spotify_client.py

# Step 2: Build features
python src/features.py

# Step 3: Train the model
python src/cluster.py

# Step 4: Launch the app
streamlit run app/streamlit_app.py
```

---

## Design Decisions (Interview Reference)

**Why K-Means over DBSCAN?**
K-Means produces a fixed number of interpretable clusters and centroid distances
are meaningful — we use them directly in the recommender to rank tracks. DBSCAN
would identify arbitrary-shaped clusters and outliers, which is more suitable for
anomaly detection than recommendation.

**Why UMAP over t-SNE?**
UMAP preserves global cluster structure (not just local neighbourhoods), is
significantly faster on large datasets, and supports a `metric` parameter — we use
cosine distance to match the similarity measure used in the recommender.

**Why StandardScaler on only tempo and loudness?**
Seven of the nine audio features are already bounded to [0, 1] and share a
comparable scale. Scaling them would distort their natural interpretation. Only
`tempo` (BPM, ~60–200) and `loudness` (dB, ~-60–0) are on different scales and
require standardisation.

**Why not a neural recommendation model?**
Interpretability. K-Means clusters are directly explainable: "this track is in the
Energetic cluster because its energy=0.9, valence=0.7, tempo=155." A neural model
would require SHAP or LIME to explain recommendations — unnecessary complexity for
a single-user system with labelled mood archetypes.

---

## Audio Features Reference

| Feature | Range | What it captures |
|---|---|---|
| energy | 0–1 | Intensity and activity |
| valence | 0–1 | Musical positivity/happiness |
| danceability | 0–1 | Rhythm suitability |
| tempo | BPM | Track speed |
| acousticness | 0–1 | Acoustic vs electronic |
| instrumentalness | 0–1 | Absence of vocals |
| liveness | 0–1 | Audience presence |
| loudness | dB | Average decibels |
| speechiness | 0–1 | Spoken word proportion |

---

## License

MIT
