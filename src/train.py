"""Train and compare mood classifiers, save the best one plus evaluation artefacts.

Usage:
    python -m src.train
"""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import DATA_PATH, FEATURES, MODEL_PATH, MOODS, REPORTS

SEED = 42


def candidate_models() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]),
        "Random Forest": Pipeline([("scale", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1))]),
        "Gradient Boosting": Pipeline([("scale", StandardScaler()), ("clf", GradientBoostingClassifier(random_state=SEED))]),
    }


def train(df: pd.DataFrame, save: bool = True) -> dict:
    X, y = df[FEATURES], df["mood"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = {}
    for name, model in candidate_models().items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro")
        cv_scores[name] = float(scores.mean())
        print(f"{name:<20} CV macro-F1 = {scores.mean():.3f} (+/- {scores.std():.3f})")

    best_name = max(cv_scores, key=cv_scores.get)
    best = candidate_models()[best_name].fit(X_train, y_train)
    preds = best.predict(X_test)
    metrics = {
        "best_model": best_name,
        "cv_macro_f1": cv_scores,
        "test_accuracy": float(accuracy_score(y_test, preds)),
        "test_macro_f1": float(f1_score(y_test, preds, average="macro")),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    print(f"\nBest model: {best_name}")
    print(classification_report(y_test, preds))

    if save:
        REPORTS.mkdir(exist_ok=True)
        MODEL_PATH.parent.mkdir(exist_ok=True)
        # refit on all data for the final artefact
        final = candidate_models()[best_name].fit(X, y)
        joblib.dump(final, MODEL_PATH)
        (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

        fig, ax = plt.subplots(figsize=(5.5, 5))
        ConfusionMatrixDisplay.from_predictions(y_test, preds, labels=MOODS, cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(f"Confusion matrix - {best_name}")
        fig.tight_layout()
        fig.savefig(REPORTS / "confusion_matrix.png", dpi=150)
        plt.close(fig)

        clf = best.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            imp = pd.Series(clf.feature_importances_, index=FEATURES).sort_values()
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(imp.index, imp.values, color="#8b5cf6")
            ax.set_title("Feature importance")
            fig.tight_layout()
            fig.savefig(REPORTS / "feature_importance.png", dpi=150)
            plt.close(fig)
        print(f"Model saved to {MODEL_PATH}")
    return metrics


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit("data/songs.csv not found. Run `python -m src.generate_data` first.")
    train(pd.read_csv(DATA_PATH))


if __name__ == "__main__":
    main()
