from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from src.utils import METRICS_DIR, MODELS_DIR, VISUALS_DIR


def evaluate_predictions(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(y_true, y_pred, target_names=["Real", "Fake"], zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def save_classification_report(report_text: str) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")


def save_confusion_matrix_plot(matrix: list[list[int]]) -> None:
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6, 5))
    heatmap = axis.imshow(matrix, cmap="Blues")
    axis.set_title("Confusion Matrix")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_xticks([0, 1], ["Real", "Fake"])
    axis.set_yticks([0, 1], ["Real", "Fake"])
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(column_index, row_index, str(value), ha="center", va="center", color="black")
    fig.colorbar(heatmap, ax=axis)
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_artifacts():
    model = joblib.load(MODELS_DIR / "fake_review_detector.pkl")
    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
    return model, vectorizer
