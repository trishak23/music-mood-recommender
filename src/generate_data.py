"""Generate a synthetic song catalogue with Spotify-style audio features and mood labels.

To use a real dataset (e.g. a Kaggle "Spotify Tracks" CSV) provide a CSV with the columns:
    track_id, title, artist, genre, mood, + the audio features listed in src/config.py
If your data has no `mood` column, label it with `label_mood_from_valence_energy` below.

Usage:
    python -m src.generate_data --songs 4000 --seed 42
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import DATA_PATH, FEATURES, MOODS

# mean of each feature per mood (0-1 features, tempo in BPM, loudness in dB)
MOOD_PROFILES = {
    "Happy":     dict(danceability=0.72, energy=0.72, valence=0.76, acousticness=0.22, instrumentalness=0.05, speechiness=0.08, liveness=0.18, tempo=120, loudness=-6.0),
    "Sad":       dict(danceability=0.40, energy=0.30, valence=0.22, acousticness=0.65, instrumentalness=0.12, speechiness=0.05, liveness=0.14, tempo=90,  loudness=-11.0),
    "Calm":      dict(danceability=0.45, energy=0.22, valence=0.50, acousticness=0.78, instrumentalness=0.45, speechiness=0.04, liveness=0.11, tempo=84,  loudness=-13.0),
    "Energetic": dict(danceability=0.62, energy=0.90, valence=0.50, acousticness=0.08, instrumentalness=0.08, speechiness=0.09, liveness=0.25, tempo=140, loudness=-4.5),
}
GENRES_BY_MOOD = {
    "Happy": (["Pop", "Dance", "Bollywood", "Indie Pop", "Funk"], [0.3, 0.25, 0.2, 0.15, 0.1]),
    "Sad": (["Acoustic", "Indie Folk", "Bollywood", "Ballad", "Blues"], [0.25, 0.2, 0.25, 0.2, 0.1]),
    "Calm": (["Ambient", "Lo-Fi", "Classical", "Acoustic", "Instrumental"], [0.25, 0.25, 0.2, 0.15, 0.15]),
    "Energetic": (["Rock", "EDM", "Hip-Hop", "Metal", "Punjabi Pop"], [0.25, 0.25, 0.2, 0.1, 0.2]),
}
FIRST = ["Aarav", "Meera", "Rohan", "Isha", "Kabir", "Zoya", "Neon", "Luna", "Arjun", "Sana", "River", "Echo", "Nova", "Dev", "Tara", "Kai"]
LAST = ["Sharma", "Rivers", "Patel", "Waves", "Khan", "Stone", "Joshi", "Fox", "Mehta", "Hart", "Das", "Lane", "Roy", "Frost", "Nair", "Vale"]
WORD_A = ["Midnight", "Golden", "Broken", "Electric", "Silent", "Dancing", "Lost", "Endless", "Neon", "Gentle", "Wild", "Fading", "Burning", "Quiet", "Rising"]
WORD_B = ["Skies", "Dreams", "Heart", "Rain", "Fire", "Echoes", "Roads", "Memories", "Waves", "Lights", "Summer", "Shadows", "Horizon", "Colors", "Story"]


def label_mood_from_valence_energy(valence: float, energy: float) -> str:
    """Russell's circumplex-style rule useful for unlabeled real datasets."""
    if energy >= 0.7:
        return "Happy" if valence >= 0.6 else "Energetic"
    if energy < 0.45:
        return "Calm" if valence >= 0.4 else "Sad"
    return "Happy" if valence >= 0.55 else "Sad"


def generate(n_songs: int = 4000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    artists = [f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(150)]
    rows = []
    for i in range(n_songs):
        mood = MOODS[i % len(MOODS)]
        prof = MOOD_PROFILES[mood]
        row = {"track_id": f"T{i:05d}", "mood": mood}
        for feat in FEATURES:
            mean = prof[feat]
            if feat == "tempo":
                val = rng.normal(mean, 18)
            elif feat == "loudness":
                val = rng.normal(mean, 2.6)
            else:
                val = rng.normal(mean, 0.14)
            row[feat] = val
        genres, p = GENRES_BY_MOOD[mood]
        row["genre"] = rng.choice(genres, p=p)
        row["artist"] = rng.choice(artists)
        row["title"] = f"{rng.choice(WORD_A)} {rng.choice(WORD_B)} {rng.integers(1, 99)}"
        rows.append(row)

    df = pd.DataFrame(rows)
    zero_one = [f for f in FEATURES if f not in ("tempo", "loudness")]
    df[zero_one] = df[zero_one].clip(0, 1)
    df["tempo"] = df["tempo"].clip(50, 210)
    df["loudness"] = df["loudness"].clip(-30, 0)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df[["track_id", "title", "artist", "genre", *FEATURES, "mood"]].round(3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--songs", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    df = generate(args.songs, args.seed)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(df):,} songs -> {DATA_PATH}")
    print(df["mood"].value_counts().to_string())


if __name__ == "__main__":
    main()
