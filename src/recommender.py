"""Mood-based and content-based song recommendation."""
from __future__ import annotations

import re

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from .config import DATA_PATH, FEATURES, MODEL_PATH, MOODS

# Simple keyword -> mood lookup so users can type how they feel
MOOD_KEYWORDS = {
    "Happy": ["happy", "joy", "good", "great", "excited", "party", "celebrate", "cheerful", "fun", "dance", "love", "awesome"],
    "Sad": ["sad", "down", "low", "cry", "lonely", "heartbreak", "heartbroken", "depressed", "miss", "blue", "upset", "hurt"],
    "Calm": ["calm", "relax", "chill", "sleep", "peace", "study", "focus", "meditate", "quiet", "stress", "anxious", "unwind"],
    "Energetic": ["energetic", "gym", "workout", "run", "pump", "hype", "power", "motivated", "fast", "drive", "angry", "intense"],
}


def mood_from_text(text: str, default: str = "Happy") -> str:
    """Return the mood whose keywords occur most often in the text."""
    words = re.findall(r"[a-z']+", text.lower())
    scores = {mood: sum(w in kws for w in words) for mood, kws in MOOD_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else default


class MoodRecommender:
    def __init__(self, songs: pd.DataFrame | None = None, model=None):
        self.songs = (songs if songs is not None else pd.read_csv(DATA_PATH)).reset_index(drop=True)
        self.model = model if model is not None else joblib.load(MODEL_PATH)

        proba = self.model.predict_proba(self.songs[FEATURES])
        self.proba = pd.DataFrame(proba, columns=self.model.classes_)
        self.songs["predicted_mood"] = self.proba.idxmax(axis=1).values

        self._scaled = StandardScaler().fit_transform(self.songs[FEATURES])

    # ------------------------------------------------------------------ mood
    def recommend(self, mood: str, n: int = 10, genre: str | None = None, max_per_artist: int = 2) -> pd.DataFrame:
        """Top-n songs most likely to match `mood` (with artist diversity)."""
        mood = mood.title()
        if mood not in self.proba.columns:
            raise ValueError(f"Unknown mood '{mood}'. Choose from {list(self.proba.columns)}")
        pool = self.songs.assign(mood_confidence=self.proba[mood].values)
        if genre:
            pool = pool[pool["genre"].str.lower() == genre.lower()]
        pool = pool.sort_values("mood_confidence", ascending=False)

        picks, per_artist = [], {}
        for idx, row in pool.iterrows():
            if per_artist.get(row["artist"], 0) >= max_per_artist:
                continue
            picks.append(idx)
            per_artist[row["artist"]] = per_artist.get(row["artist"], 0) + 1
            if len(picks) == n:
                break
        cols = ["track_id", "title", "artist", "genre", "mood_confidence"]
        return pool.loc[picks, cols].reset_index(drop=True).round({"mood_confidence": 3})

    def recommend_from_text(self, text: str, n: int = 10, genre: str | None = None) -> tuple[str, pd.DataFrame]:
        mood = mood_from_text(text)
        return mood, self.recommend(mood, n, genre)

    # ------------------------------------------------------------ similarity
    def find_song(self, query: str) -> int:
        """Find a song's row index by track_id or (partial) title."""
        by_id = self.songs.index[self.songs["track_id"] == query]
        if len(by_id):
            return int(by_id[0])
        by_title = self.songs.index[self.songs["title"].str.contains(query, case=False, regex=False)]
        if not len(by_title):
            raise KeyError(f"No song matches '{query}'")
        return int(by_title[0])

    def similar_songs(self, query: str, n: int = 10) -> pd.DataFrame:
        """Songs with the closest audio profile (cosine similarity on scaled features)."""
        idx = self.find_song(query)
        sims = cosine_similarity(self._scaled[idx : idx + 1], self._scaled).ravel()
        sims[idx] = -1  # exclude the song itself
        top = np.argsort(sims)[::-1][:n]
        out = self.songs.loc[top, ["track_id", "title", "artist", "genre", "predicted_mood"]].copy()
        out["similarity"] = sims[top].round(3)
        return out.reset_index(drop=True)
