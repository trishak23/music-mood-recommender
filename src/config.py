"""Shared constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "songs.csv"
MODEL_PATH = ROOT / "models" / "mood_classifier.joblib"
REPORTS = ROOT / "reports"

# Audio features (same names as the Spotify "Tracks" datasets on Kaggle)
FEATURES = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "liveness",
    "tempo",
    "loudness",
]
MOODS = ["Happy", "Sad", "Calm", "Energetic"]
