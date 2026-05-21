from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from src.utils import CONFIG


def split_dataset(frame: pd.DataFrame):
    features = frame[
        [
            "cleaned_review_text",
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=CONFIG.test_size, random_state=CONFIG.random_state)
        train_index, test_index = next(splitter.split(features, target))
        X_train = features.iloc[train_index].reset_index(drop=True)
        X_test = features.iloc[test_index].reset_index(drop=True)
        y_train = target.iloc[train_index].reset_index(drop=True)
        y_test = target.iloc[test_index].reset_index(drop=True)
        return X_train, X_test, y_train, y_test
            "verified_purchase",
            "review_length",
            "sentiment_score",
            "suspicious_word_count",
            "uppercase_word_count",
            "exclamation_count",
        ]
    ].copy()
    target = frame["fake_review"].map({"Real": 0, "Fake": 1})
    return train_test_split(features, target, test_size=CONFIG.test_size, random_state=CONFIG.random_state, stratify=target)


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=CONFIG.max_features,
        stop_words="english",
        ngram_range=CONFIG.ngram_range,
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        smooth_idf=True,
    )


def scale_numeric_features(train_frame: pd.DataFrame, test_frame: pd.DataFrame, columns: list[str]):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_frame[columns].astype(float))
    test_scaled = scaler.transform(test_frame[columns].astype(float))
    return train_scaled, test_scaled, scaler
