"""Streamlit app: pick a mood (or describe how you feel) and get a playlist.

Run:  streamlit run app.py
"""
import streamlit as st

from src.config import DATA_PATH, MODEL_PATH, MOODS
from src.recommender import MoodRecommender, mood_from_text

st.set_page_config(page_title="Music Mood Recommender", page_icon="🎧", layout="wide")
st.title("🎧 Music Mood Recommender")
st.caption("A classifier predicts the mood of each track from its audio features; recommendations are ranked by mood confidence.")

if not (DATA_PATH.exists() and MODEL_PATH.exists()):
    st.error("Data or model missing. Run `python -m src.generate_data` and `python -m src.train` first.")
    st.stop()


@st.cache_resource(show_spinner="Loading model and scoring songs...")
def load_recommender() -> MoodRecommender:
    return MoodRecommender()


rec = load_recommender()
EMOJI = {"Happy": "😄", "Sad": "😢", "Calm": "😌", "Energetic": "⚡"}

with st.sidebar:
    st.header("Your vibe")
    mode = st.radio("Choose by", ["Mood", "Describe how I feel"])
    if mode == "Mood":
        mood = st.selectbox("Mood", MOODS, format_func=lambda m: f"{EMOJI[m]} {m}")
    else:
        text = st.text_input("How are you feeling?", "need to relax and focus")
        mood = None
    genre = st.selectbox("Genre (optional)", ["Any"] + sorted(rec.songs["genre"].unique()))
    n = st.slider("Number of songs", 5, 30, 10)

genre = None if genre == "Any" else genre
if mode != "Mood":
    mood = mood_from_text(text)
    st.info(f"Detected mood: {EMOJI[mood]} **{mood}**")

tab_playlist, tab_similar, tab_explore = st.tabs(["Playlist", "Find similar songs", "Explore data"])

with tab_playlist:
    st.subheader(f"{EMOJI[mood]} {mood} playlist")
    st.dataframe(rec.recommend(mood, n, genre), use_container_width=True, hide_index=True)

with tab_similar:
    query = st.text_input("Song title (or part of it) or track id", "Golden")
    if query:
        try:
            st.dataframe(rec.similar_songs(query, n), use_container_width=True, hide_index=True)
        except KeyError as err:
            st.warning(str(err))

with tab_explore:
    st.write("Average audio features per predicted mood")
    avg = rec.songs.groupby("predicted_mood")[["danceability", "energy", "valence", "acousticness"]].mean()
    st.bar_chart(avg)
    st.scatter_chart(rec.songs.sample(min(1500, len(rec.songs)), random_state=0), x="valence", y="energy", color="predicted_mood")
