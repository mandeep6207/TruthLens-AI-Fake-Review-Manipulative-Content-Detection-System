from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import csr_matrix, hstack
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud

from src.evaluate_model import evaluate_predictions, save_classification_report, save_confusion_matrix_plot
from src.feature_engineering import build_vectorizer, scale_numeric_features, split_dataset
from src.preprocessing import build_clean_dataset, generate_synthetic_reviews
from src.utils import CONFIG, DATA_DIR, MODELS_DIR, REPORTS_DIR, VISUALS_DIR, ensure_directories, save_json

try:
    import nltk

    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:  # pragma: no cover - runtime bootstrap
    import nltk

    nltk.download("vader_lexicon", quiet=True)


def generate_visuals(frame: pd.DataFrame, feature_names=None, importances=None) -> None:
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    label_counts = frame["fake_review"].value_counts().reindex(["Real", "Fake"])

    fig, axis = plt.subplots(figsize=(7, 5))
    axis.bar(label_counts.index, label_counts.values, color=["#2E86AB", "#D1495B"])
    axis.set_title("Review Distribution")
    axis.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / "review_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 7))
    axis.pie(label_counts.values, labels=label_counts.index, autopct="%1.1f%%", colors=["#2E86AB", "#D1495B"], startangle=90)
    axis.set_title("Fake vs Real Pie Chart")
    fig.savefig(VISUALS_DIR / "fake_vs_real_pie.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    real_text = " ".join(frame.loc[frame["fake_review"] == "Real", "cleaned_review_text"].astype(str).tolist())
    fake_text = " ".join(frame.loc[frame["fake_review"] == "Fake", "cleaned_review_text"].astype(str).tolist())
    for text, filename, title in [
        (real_text, "wordcloud_real.png", "WordCloud for Real Reviews"),
        (fake_text, "wordcloud_fake.png", "WordCloud for Fake Reviews"),
    ]:
        wc = WordCloud(width=1200, height=700, background_color="white", colormap="viridis").generate(text)
        fig, axis = plt.subplots(figsize=(12, 7))
        axis.imshow(wc, interpolation="bilinear")
        axis.axis("off")
        axis.set_title(title)
        fig.savefig(VISUALS_DIR / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    sns.histplot(data=frame, x="sentiment_score", hue="fake_review", kde=True, bins=40, palette=["#2E86AB", "#D1495B"], ax=axis)
    axis.set_title("Sentiment Distribution")
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / "sentiment_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    sns.histplot(data=frame, x="sentiment_intensity", hue="fake_review", kde=True, bins=40, palette=["#0F766E", "#B91C1C"], ax=axis)
    axis.set_title("Sentiment Intensity Analysis")
    axis.set_xlabel("VADER compound score")
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / "sentiment_intensity_analysis.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=frame, x="fake_review", y="review_length", hue="fake_review", palette=["#2E86AB", "#D1495B"], ax=axis, legend=False)
    axis.set_title("Review Length Analysis")
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / "review_length_analysis.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    sns.violinplot(data=frame, x="fake_review", y="review_length", inner="quartile", palette=["#2E86AB", "#D1495B"], ax=axis)
    sns.stripplot(data=frame.sample(n=min(len(frame), 500), random_state=CONFIG.random_state), x="fake_review", y="review_length", color="black", alpha=0.08, ax=axis)
    axis.set_title("Advanced Review Length Distribution")
    axis.set_xlabel("Label")
    axis.set_ylabel("Token count")
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / "review_length_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    numeric_columns = ["rating", "verified_purchase", "review_length", "sentiment_score", "suspicious_word_count", "uppercase_word_count", "exclamation_count"]
    fig, axis = plt.subplots(figsize=(9, 7))
    correlation_matrix = frame[numeric_columns].corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=axis,
    )
    axis.set_title("Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / "correlation_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    if feature_names is not None and importances is not None:
        top_indices = np.argsort(importances)[-20:]
        top_features = np.array(feature_names)[top_indices]
        top_values = np.array(importances)[top_indices]
        fig, axis = plt.subplots(figsize=(10, 7))
        axis.barh(top_features, top_values, color="#2E86AB")
        axis.set_title("Feature Importance Plot")
        fig.tight_layout()
        fig.savefig(VISUALS_DIR / "feature_importance.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def train_and_evaluate():
    ensure_directories()
    dataset = generate_synthetic_reviews()
    dataset.to_csv(DATA_DIR / "reviews_dataset.csv", index=False)
    cleaned = build_clean_dataset()
    generate_visuals(cleaned)

    X_train, X_test, y_train, y_test = split_dataset(cleaned)
    text_train = X_train["cleaned_review_text"].astype(str)
    text_test = X_test["cleaned_review_text"].astype(str)
    numeric_features = ["rating", "verified_purchase", "review_length", "sentiment_score", "suspicious_word_count"]
    numeric_train, numeric_test, _ = scale_numeric_features(X_train, X_test, numeric_features)

    vectorizer = build_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(text_train)
    X_test_tfidf = vectorizer.transform(text_test)
    X_train_combined = hstack([X_train_tfidf, csr_matrix(numeric_train)])
    X_test_combined = hstack([X_test_tfidf, csr_matrix(numeric_test)])
    combined_feature_names = list(vectorizer.get_feature_names_out()) + numeric_features

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=CONFIG.random_state, C=0.7, solver="liblinear"),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.7),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=180,
            random_state=CONFIG.random_state,
            class_weight="balanced_subsample",
            n_jobs=-1,
            max_depth=10,
            min_samples_leaf=6,
            min_samples_split=10,
            max_features="sqrt",
        ),
    }

    results = {}
    trained_models = {}
    for model_name, model in models.items():
        if model_name == "Random Forest Classifier":
            model.fit(X_train_combined.toarray(), y_train)
            predictions = model.predict(X_test_combined.toarray())
        else:
            model.fit(X_train_combined, y_train)
            predictions = model.predict(X_test_combined)
        metrics = evaluate_predictions(y_test, predictions)
        results[model_name] = {key: value for key, value in metrics.items() if key in {"accuracy", "precision", "recall", "weighted_f1"}}
        trained_models[model_name] = (model, predictions, metrics)

    best_model_name = max(results, key=lambda key: results[key]["weighted_f1"])
    best_model, best_predictions, best_metrics = trained_models[best_model_name]

    joblib.dump(best_model, MODELS_DIR / "fake_review_detector.pkl")
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")

    save_classification_report(best_metrics["classification_report"])
    save_confusion_matrix_plot(best_metrics["confusion_matrix"])

    rf_model = RandomForestClassifier(
        n_estimators=180,
        random_state=CONFIG.random_state,
        class_weight="balanced_subsample",
        n_jobs=-1,
        max_depth=10,
        min_samples_leaf=6,
        min_samples_split=10,
        max_features="sqrt",
    )
    rf_model.fit(X_train_combined.toarray(), y_train)
    generate_visuals(cleaned, combined_feature_names, rf_model.feature_importances_)

    cleaned.to_csv(DATA_DIR / "cleaned_reviews.csv", index=False)

    save_json(REPORTS_DIR / "model_metrics.json", {
        "best_model": best_model_name,
        "model_comparison": results,
        "best_model_report": best_metrics,
    })

    report_lines = [
        "# TruthLens AI Project Report",
        "",
        f"Best model: {best_model_name}",
        "",
        "## Model Comparison",
        json.dumps(results, indent=2),
        "",
        "## Key Insights",
        "- Fake reviews are intentionally more repetitive, promotional, and exclamation-heavy.",
        "- Verified purchase status and suspicious word counts are strong behavioral signals.",
        "- The TF-IDF model captures strong lexical separation between real and fake reviews.",
    ]
    (REPORTS_DIR / "project_report.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    train_and_evaluate()
