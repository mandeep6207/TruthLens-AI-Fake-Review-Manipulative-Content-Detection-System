from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from src.utils import CONFIG, DATA_DIR, ensure_directories

try:
    import nltk

    nltk.data.find("corpora/stopwords")
except LookupError:  # pragma: no cover - runtime bootstrap
    import nltk

    nltk.download("stopwords", quiet=True)


STEMMER = PorterStemmer()
STOPWORDS = set(stopwords.words("english"))
SUSPICIOUS_TERMS = {
    "buy now",
    "limited time",
    "best ever",
    "highly recommend",
    "amazing product",
    "life changing",
    "must have",
    "five stars",
    "incredible",
    "perfect",
    "guaranteed",
    "discount",
    "promo",
    "sponsored",
    "paid review",
    "unbelievable",
    "fantastic",
    "outstanding",
    "top quality",
}


REAL_OPENERS = [
    "I bought this last month and",
    "After a few weeks of use",
    "My experience has been",
    "I have used this product for",
    "For the price, this item",
    "The delivery was on time and",
    "I wanted something reliable and",
    "This has been part of my daily routine and",
]

REAL_ADJECTIVES = [
    "useful",
    "solid",
    "practical",
    "fair",
    "consistent",
    "comfortable",
    "noticeable",
    "efficient",
    "simple",
    "balanced",
]

REAL_LIMITATIONS = [
    "the battery life could be better",
    "the setup took a little longer than expected",
    "the packaging was acceptable but not premium",
    "I wish the instructions were clearer",
    "it works well, although it is not perfect",
    "there are a few minor trade-offs",
]

FAKE_OPENERS = [
    "Absolutely perfect!!!",
    "Best purchase ever!!!",
    "This product is unbelievable!!!",
    "I am obsessed with this amazing item!!!",
    "Five stars all the way!!!",
    "Must buy now!!!",
    "This changed everything!!!",
    "Superb deal!!!",
]

FAKE_MARKETING = [
    "limited time offer",
    "best ever",
    "highly recommend",
    "buy now",
    "life changing",
    "top quality",
    "five stars",
    "must have",
    "incredible",
    "discount",
    "promo",
    "sponsored",
    "unbelievable",
    "fantastic",
]


def _sentiment_score(text: str) -> float:
    positive_hits = sum(term in text.lower() for term in ["good", "great", "excellent", "love", "reliable", "useful", "solid", "comfortable"])
    negative_hits = sum(term in text.lower() for term in ["bad", "poor", "broken", "slow", "issue", "problem", "frustrating", "cheap"])
    score = (positive_hits - negative_hits) / max(len(text.split()), 1)
    return float(np.clip(score, -1.0, 1.0))


def generate_synthetic_reviews(sample_size: int = CONFIG.sample_size, random_state: int = CONFIG.random_state) -> pd.DataFrame:
    rng = random.Random(random_state)
    np_rng = np.random.default_rng(random_state)
    records = []
    fake_count = sample_size // 2
    real_count = sample_size - fake_count

    for index in range(real_count):
        opener = rng.choice(REAL_OPENERS)
        adjective = rng.choice(REAL_ADJECTIVES)
        limitation = rng.choice(REAL_LIMITATIONS)
        sentiment_tail = rng.choice([
            "I would still recommend it for everyday use.",
            "Overall it feels dependable and practical.",
            "It does what I need without much fuss.",
            "I would buy it again if I needed another one.",
            "The value is decent for the price point.",
        ])
        text = f"{opener} {adjective} overall. {limitation}. {sentiment_tail}"
        if index % 4 == 0:
            text += " The finish is decent and the performance stays consistent."
        rating = int(np_rng.choice([3, 4, 5], p=[0.18, 0.42, 0.40]))
        verified_purchase = int(np_rng.choice([0, 1], p=[0.08, 0.92]))
        records.append({"review_text": text, "rating": rating, "verified_purchase": verified_purchase, "fake_review": "Real"})

    for index in range(fake_count):
        opener = rng.choice(FAKE_OPENERS)
        marketing = rng.sample(FAKE_MARKETING, k=4)
        repeated = rng.choice([
            "Amazing amazing amazing product",
            "Perfect perfect perfect quality",
            "Best best best ever",
            "Absolutely incredible and fantastic",
        ])
        text = (
            f"{opener} {repeated}. "
            f"{marketing[0].title()} {marketing[1]} and {marketing[2]} {marketing[3]}! "
            "This is the most fantastic purchase I have ever made and I highly recommend it to everyone."
        )
        if index % 3 == 0:
            text += " BUY NOW if you want the best deal!!!"
        rating = int(np_rng.choice([1, 4, 5], p=[0.06, 0.28, 0.66]))
        verified_purchase = int(np_rng.choice([0, 1], p=[0.82, 0.18]))
        records.append({"review_text": text, "rating": rating, "verified_purchase": verified_purchase, "fake_review": "Fake"})

    frame = pd.DataFrame.from_records(records)
    frame = frame.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    frame["review_length"] = frame["review_text"].str.split().str.len()
    frame["sentiment_score"] = frame["review_text"].apply(_sentiment_score)
    frame["suspicious_word_count"] = frame["review_text"].str.lower().apply(lambda value: sum(term in value for term in SUSPICIOUS_TERMS))
    frame["uppercase_word_count"] = frame["review_text"].apply(lambda value: sum(token.isupper() and len(token) > 1 for token in re.findall(r"\b\w+\b", value)))
    frame["exclamation_count"] = frame["review_text"].str.count(r"!")
    return frame


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [STEMMER.stem(token) for token in text.split() if token not in STOPWORDS and len(token) > 1]
    return " ".join(tokens)


def build_clean_dataset(source_path: Path | None = None, output_path: Path | None = None) -> pd.DataFrame:
    ensure_directories()
    source_path = source_path or DATA_DIR / "reviews_dataset.csv"
    output_path = output_path or DATA_DIR / "cleaned_reviews.csv"
    frame = pd.read_csv(source_path)
    frame["cleaned_review_text"] = frame["review_text"].astype(str).apply(preprocess_text)
    frame["review_length"] = frame["review_text"].astype(str).str.split().str.len()
    frame.to_csv(output_path, index=False)
    return frame


if __name__ == "__main__":
    ensure_directories()
    dataset = generate_synthetic_reviews()
    dataset.to_csv(DATA_DIR / "reviews_dataset.csv", index=False)
    build_clean_dataset()
