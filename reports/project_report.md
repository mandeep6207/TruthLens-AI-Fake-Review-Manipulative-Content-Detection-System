# TruthLens AI Project Report

Best model: Logistic Regression

## Model Comparison
{
  "Logistic Regression": {
    "accuracy": 0.96125,
    "precision": 0.9613208753219474,
    "recall": 0.96125,
    "weighted_f1": 0.9612478806971159
  },
  "Multinomial Naive Bayes": {
    "accuracy": 0.96125,
    "precision": 0.9613208753219474,
    "recall": 0.96125,
    "weighted_f1": 0.9612478806971159
  },
  "Random Forest Classifier": {
    "accuracy": 0.96125,
    "precision": 0.9613208753219474,
    "recall": 0.96125,
    "weighted_f1": 0.9612478806971159
  }
}

## Key Insights
- Fake reviews are intentionally more repetitive, promotional, and exclamation-heavy.
- Verified purchase status and suspicious word counts are strong behavioral signals.
- The TF-IDF model captures strong lexical separation between real and fake reviews.