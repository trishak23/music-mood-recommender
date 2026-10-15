"""Command-line interface.

Examples:
    python -m src.cli --mood happy -n 5
    python -m src.cli --text "feeling low after a long day" --genre Acoustic
    python -m src.cli --similar-to "Golden Skies"
"""
from __future__ import annotations

import argparse

from .recommender import MoodRecommender


def main() -> None:
    parser = argparse.ArgumentParser(description="Music mood recommendation system")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mood", help="Happy | Sad | Calm | Energetic")
    group.add_argument("--text", help="Describe how you feel, e.g. 'need to focus and relax'")
    group.add_argument("--similar-to", help="Track id or part of a title")
    parser.add_argument("-n", type=int, default=10, help="Number of songs")
    parser.add_argument("--genre", help="Optional genre filter")
    args = parser.parse_args()

    rec = MoodRecommender()
    if args.mood:
        print(f"\nTop {args.n} {args.mood.title()} songs:\n")
        print(rec.recommend(args.mood, args.n, args.genre).to_string(index=False))
    elif args.text:
        mood, result = rec.recommend_from_text(args.text, args.n, args.genre)
        print(f"\nDetected mood: {mood}\n")
        print(result.to_string(index=False))
    else:
        print(f"\nSongs similar to '{args.similar_to}':\n")
        print(rec.similar_songs(args.similar_to, args.n).to_string(index=False))


if __name__ == "__main__":
    main()
