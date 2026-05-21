from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
VISUALS_DIR = ROOT / "visuals"
REPORTS_DIR = ROOT / "reports"
METRICS_DIR = ROOT / "metrics"
NOTEBOOKS_DIR = ROOT / "notebooks"


@dataclass(frozen=True)
class AppConfig:
    random_state: int = 42
    sample_size: int = 4000
    max_features: int = 3000
    ngram_range: tuple[int, int] = (1, 2)
    test_size: float = 0.2


CONFIG = AppConfig()


def ensure_directories() -> None:
    for directory in [DATA_DIR, MODELS_DIR, VISUALS_DIR, REPORTS_DIR, METRICS_DIR, NOTEBOOKS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)


def slugify_tokens(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def batched(iterable: Iterable, size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
