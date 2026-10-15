import pytest

from src.config import FEATURES, MOODS
from src.generate_data import generate, label_mood_from_valence_energy
from src.recommender import MoodRecommender, mood_from_text
from src.train import candidate_models


@pytest.fixture(scope="module")
def songs():
    return generate(n_songs=800, seed=7)


@pytest.fixture(scope="module")
def recommender(songs):
    model = candidate_models()["Logistic Regression"].fit(songs[FEATURES], songs["mood"])
    return MoodRecommender(songs=songs.copy(), model=model)


def test_generated_data_is_valid(songs):
    assert set(songs["mood"]) == set(MOODS)
    assert songs[FEATURES].notna().all().all()
    assert songs["valence"].between(0, 1).all()
    assert songs["track_id"].is_unique


def test_classifier_accuracy(recommender, songs):
    acc = (recommender.songs["predicted_mood"] == songs["mood"]).mean()
    assert acc > 0.85


def test_text_to_mood():
    cases = {
        "I am so happy and want to party": "Happy",
        "feeling lonely and sad": "Sad",
        "need to relax and study": "Calm",
        "going to the gym for a workout": "Energetic",
        "zzz qqq": "Happy",  # default when no keyword matches
    }
    for text, expected in cases.items():
        assert mood_from_text(text) == expected


def test_recommend_returns_requested_mood(recommender):
    result = recommender.recommend("Calm", n=10)
    assert len(result) == 10
    merged = result.merge(recommender.songs[["track_id", "predicted_mood"]], on="track_id")
    assert (merged["predicted_mood"] == "Calm").all()
    assert result["artist"].value_counts().max() <= 2


def test_recommend_genre_filter(recommender):
    genre = recommender.songs["genre"].iloc[0]
    result = recommender.recommend("Happy", n=5, genre=genre)
    assert (result["genre"] == genre).all()


def test_unknown_mood_raises(recommender):
    with pytest.raises(ValueError):
        recommender.recommend("Angry")


def test_similar_songs_excludes_itself(recommender):
    track = recommender.songs["track_id"].iloc[0]
    result = recommender.similar_songs(track, n=5)
    assert track not in set(result["track_id"])
    assert result["similarity"].is_monotonic_decreasing


def test_label_rule():
    assert label_mood_from_valence_energy(0.9, 0.8) == "Happy"
    assert label_mood_from_valence_energy(0.2, 0.2) == "Sad"
