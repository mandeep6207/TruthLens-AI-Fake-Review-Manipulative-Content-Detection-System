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
NEGATION_WORDS = {"no", "not", "nor", "never", "without"}
DOMAIN_STOPWORDS = {
    "product",
    "item",
    "purchase",
    "review",
    "buy",
    "bought",
    "use",
    "used",
    "thing",
    "things",
}
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

SARCASM_SNIPPETS = [
    "yeah, because everything is always perfect",
    "sure, this is the miracle product everyone talks about",
    "totally not overhyped at all",
    "what a surprise, it is actually decent",
    "absolutely amazing, if you ignore the obvious issues",
]

SARCASTIC_REAL_PHRASES = [
    "Sure, it is perfect in the same way traffic is relaxing.",
    "Apparently every product has to be a life changer now.",
    "The hype was loud, but the item itself was just okay.",
    "I guess this is what premium means these days.",
]

SARCASTIC_FAKE_PHRASES = [
    "Yeah, because nothing says honest review like shouting BUY NOW.",
    "Totally organic praise, definitely not an ad.",
    "The miracle product somehow needs one more five-star comment.",
    "Sure, this was written by a real customer and not a promo script.",
]

TYPO_MAP = {
    "excellent": "excellant",
    "product": "prodcut",
    "quality": "quailty",
    "recommend": "reccomend",
    "battery": "battrey",
    "delivery": "delievery",
    "amazing": "amazng",
    "service": "servcie",
}

TOKEN_REPLACEMENTS = {
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'m": " am",
    "'ll": " will",
    "'ve": " have",
}


def cleanup_review_tokens(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\b(\w)\1{2,}\b", r"\1\1", text)
    text = re.sub(r"\b(?:lol|lmao|omg|uhh+|hmm+|meh)\b", " ", text)
    text = re.sub(r"\b\w{1}\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def filter_review_tokens(tokens: list[str]) -> list[str]:
    filtered_tokens = []
    for token in tokens:
        if token in NEGATION_WORDS:
            filtered_tokens.append(token)
            continue
        if token in STOPWORDS or token in DOMAIN_STOPWORDS:
            continue
        if len(token) <= 1:
            continue
        filtered_tokens.append(token)
    return filtered_tokens

NEUTRAL_PHRASES = [
    "It arrived as expected and does the job.",
    "I have no strong feelings either way.",
    "The experience was fine overall.",
    "Nothing major stood out during use.",
    "It is a standard product with a few trade-offs.",
]

REAL_DETAIL_PHRASES = [
    "The packaging was ordinary but acceptable.",
    "Setup took a few minutes longer than I expected.",
    "It works well for everyday use and feels reasonably sturdy.",
    "I noticed a couple of small quirks, but nothing deal-breaking.",
    "The value is decent even though it is not the cheapest option.",
]

FAKE_PROMO_PHRASES = [
    "This is the best deal I have seen this year.",
    "I would definitely tell everyone to buy it now.",
    "The product feels premium and absolutely worth the hype.",
    "It is a must-have if you want instant results.",
    "I keep recommending it because the value is unbelievable.",
]

MIXED_PHRASES = [
    "I like parts of it, but the finish could be better.",
    "It is useful, although not quite as amazing as advertised.",
    "The core idea works, but the details are a bit messy.",
    "I would rate it higher if the instructions were clearer.",
    "There is a good product here, just not a perfect one.",
]

REAL_OPENERS = [
    "I bought this last month and",
    "After a few weeks of use,",
    "My experience has been",
    "I have used this product for",
    "For the price, this item is",
    "The delivery was on time and",
    "I wanted something reliable and",
    "This has been part of my daily routine and",
]

FAKE_OPENERS = [
    "Absolutely perfect",
    "Best purchase ever",
    "This product is unbelievable",
    "I am obsessed with this item",
    "Five stars all the way",
    "Must buy now",
    "This changed everything",
    "Superb deal",
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
    "mixed",
    "average",
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
    "worth the hype",
]


def _inject_typos(text: str, rng: random.Random, probability: float = 0.18) -> str:
    mutated = text
    for source, replacement in TYPO_MAP.items():
        if source in mutated.lower() and rng.random() < probability:
            mutated = re.sub(source, replacement, mutated, flags=re.IGNORECASE)
    if rng.random() < 0.12:
        mutated = mutated.replace("ing ", "in ")
    if rng.random() < 0.1:
        mutated = mutated.replace(" and ", " & ")
    return mutated


def _shape_text(text: str, rng: random.Random) -> str:
    if rng.random() < 0.2:
        text = text.replace(".", "")
    if rng.random() < 0.22:
        text = text.rstrip() + rng.choice(["!", "!!", ""])
    if rng.random() < 0.15:
        text = text + " " + rng.choice(SARCASM_SNIPPETS)
    if rng.random() < 0.14:
        text = _inject_typos(text, rng)
    return text


def normalize_review_text(text: str) -> str:
    normalized = str(text).lower()
    normalized = normalized.replace("’", "'").replace("‘", "'")
    normalized = re.sub(r"<[^>]+>", " ", normalized)
    for source, replacement in TOKEN_REPLACEMENTS.items():
        normalized = normalized.replace(source, replacement)
    normalized = re.sub(r"(.)\1{2,}", r"\1\1", normalized)
    normalized = re.sub(r"[^a-z0-9\s!?.]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


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
    positive_hits = sum(term in text.lower() for term in ["good", "great", "excellent", "love", "reliable", "useful", "solid", "comfortable", "premium", "worth"])
    negative_hits = sum(term in text.lower() for term in ["bad", "poor", "broken", "slow", "issue", "problem", "frustrating", "cheap", "messy", "awkward"])
    score = (positive_hits - negative_hits) / max(len(text.split()), 1)
    return float(np.clip(score, -1.0, 1.0))


def generate_synthetic_reviews(sample_size: int = CONFIG.sample_size, random_state: int = CONFIG.random_state) -> pd.DataFrame:
    rng = random.Random(random_state)
    np_rng = np.random.default_rng(random_state)
    records = []
    fake_count = sample_size // 2
    real_count = sample_size - fake_count

    real_profile_weights = [0.28, 0.26, 0.26, 0.2]
    fake_profile_weights = [0.24, 0.28, 0.3, 0.18]

    for index in range(real_count):
        profile = np_rng.choice(["balanced", "mixed", "neutral", "skeptical"], p=real_profile_weights)
        opener = rng.choice(REAL_OPENERS)
        adjective = rng.choice(REAL_ADJECTIVES)
        detail = rng.choice(REAL_DETAIL_PHRASES)
        neutral = rng.choice(NEUTRAL_PHRASES)
        mixed = rng.choice(MIXED_PHRASES)
        sarcasm = rng.choice(SARCASM_SNIPPETS)
        support_phrase = rng.choice([
            "I would still recommend it for everyday use.",
            "Overall it feels dependable and practical.",
            "It does what I need without much fuss.",
            "I would buy it again if I needed another one.",
            "The value is decent for the price point.",
            "It is okay, but the hype is a little much.",
        ])
        if profile == "balanced":
            text = f"{opener} {adjective} overall. {detail} {support_phrase}"
        elif profile == "mixed":
            text = f"{opener} {adjective} overall, though {mixed.lower()} {detail} {support_phrase}"
        elif profile == "neutral":
            text = f"{opener} {neutral.lower()} {detail} {support_phrase}"
        else:
            text = f"{opener} {adjective} at first, but {mixed.lower()} {neutral.lower()} {sarcasm}. {support_phrase}"
        if index % 5 == 0:
            text += " The finish is decent, and the performance stays consistent."
        if rng.random() < 0.08:
            text += " best ever? not really, just usable."
        if rng.random() < 0.28:
            text += " " + rng.choice([
                "The promo wording felt a little excessive, but the item still worked.",
                "It almost sounded sponsored in places, though the product was fine.",
                "A few phrases were too polished for my taste.",
            ])
        if rng.random() < 0.18:
            text += " " + rng.choice(SARCASTIC_REAL_PHRASES)
        text = _shape_text(text, rng)
        rating = int(np_rng.choice([1, 2, 3, 4, 5], p=[0.08, 0.15, 0.25, 0.28, 0.24]))
        if profile == "neutral":
            rating = int(np_rng.choice([2, 3, 4], p=[0.2, 0.5, 0.3]))
        verified_purchase = int(np_rng.choice([0, 1], p=[0.26, 0.74]))
        if rng.random() < 0.14:
            verified_purchase = 0
        records.append({"review_text": text, "rating": rating, "verified_purchase": verified_purchase, "fake_review": "Real"})

    for index in range(fake_count):
        profile = np_rng.choice(["promo", "mixed", "realistic", "spammy"], p=fake_profile_weights)
        opener = rng.choice(FAKE_OPENERS)
        marketing = rng.sample(FAKE_MARKETING, k=4)
        repeated = rng.choice([
            "Amazing product with great results",
            "Perfect quality and excellent value",
            "Best choice I have made in a while",
            "Absolutely incredible and surprisingly useful",
        ])
        promo_sentence = rng.choice(FAKE_PROMO_PHRASES)
        realistic_sentence = rng.choice(REAL_DETAIL_PHRASES)
        neutral = rng.choice(NEUTRAL_PHRASES)
        mixed = rng.choice(MIXED_PHRASES)
        if profile == "promo":
            text = (
                f"{opener}. {repeated}. "
                f"{marketing[0].title()} {marketing[1]} and {marketing[2]} {marketing[3]}. "
                f"{promo_sentence}"
            )
        elif profile == "mixed":
            text = (
                f"{opener}, but {mixed.lower()} {promo_sentence.lower()} "
                f"{realistic_sentence} {marketing[0]} if you want value."
            )
        elif profile == "realistic":
            text = (
                f"{opener}. {realistic_sentence} {neutral.lower()} "
                f"Still, I would highly recommend it for most people."
            )
        else:
            text = (
                f"{opener}. {repeated.lower()} {promo_sentence.lower()} "
                f"{(sarcasm := rng.choice(SARCASM_SNIPPETS)) if True else ''} buy now for the best deal."
            )
        if index % 4 == 0:
            text += " This is still worth it and I would buy again."
        if rng.random() < 0.22:
            text += " maybe it is not perfect, but the promo is real."
        if rng.random() < 0.22:
            text += " " + rng.choice(SARCASTIC_FAKE_PHRASES)
        text = _shape_text(text, rng)
        rating = int(np_rng.choice([1, 2, 3, 4, 5], p=[0.1, 0.16, 0.24, 0.26, 0.24]))
        if profile == "realistic":
            rating = int(np_rng.choice([3, 4, 5], p=[0.26, 0.42, 0.32]))
        verified_purchase = int(np_rng.choice([0, 1], p=[0.57, 0.43]))
        if rng.random() < 0.12:
            verified_purchase = 1
        records.append({"review_text": text, "rating": rating, "verified_purchase": verified_purchase, "fake_review": "Fake"})

    frame = pd.DataFrame.from_records(records)
    frame = frame.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    noisy_index = frame.sample(frac=0.04, random_state=random_state + 7).index
    frame.loc[noisy_index, "fake_review"] = frame.loc[noisy_index, "fake_review"].map({"Real": "Fake", "Fake": "Real"})

    frame["review_length"] = frame["review_text"].str.split().str.len()
    frame["sentiment_score"] = frame["review_text"].apply(_sentiment_score)
    frame["suspicious_word_count"] = frame["review_text"].str.lower().apply(lambda value: sum(term in value for term in SUSPICIOUS_TERMS))
    frame["uppercase_word_count"] = frame["review_text"].apply(lambda value: sum(token.isupper() and len(token) > 1 for token in re.findall(r"\b\w+\b", value)))
    frame["exclamation_count"] = frame["review_text"].str.count(r"!")
    frame["suspicious_word_count"] = frame["suspicious_word_count"].clip(upper=5)
    frame["uppercase_word_count"] = frame["uppercase_word_count"].clip(upper=4)
    frame["exclamation_count"] = frame["exclamation_count"].clip(upper=8)
    return frame


def preprocess_text(text: str) -> str:
    text = normalize_review_text(text)
    text = cleanup_review_tokens(text)
    tokens = [STEMMER.stem(token) for token in filter_review_tokens(text.split())]
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
