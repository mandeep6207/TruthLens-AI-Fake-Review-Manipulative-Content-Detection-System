# TruthLens AI Project Report

Best model: Logistic Regression

## Model Comparison
{
  "Logistic Regression": {
    "accuracy": 0.9624060150375939,
    "precision": 0.9625106022052586,
    "recall": 0.9624060150375939,
    "weighted_f1": 0.9624038896426955
  },
  "Multinomial Naive Bayes": {
    "accuracy": 0.9624060150375939,
    "precision": 0.9625106022052586,
    "recall": 0.9624060150375939,
    "weighted_f1": 0.9624038896426955
  },
  "Random Forest Classifier": {
    "accuracy": 0.9624060150375939,
    "precision": 0.9625106022052586,
    "recall": 0.9624060150375939,
    "weighted_f1": 0.9624038896426955
  }
}

## Key Insights
- Fake reviews are intentionally more repetitive, promotional, and exclamation-heavy.
- Verified purchase status and suspicious word counts are strong behavioral signals.
- The TF-IDF model captures strong lexical separation between real and fake reviews.
- Class balance check: True
- Average model confidence: 0.918
- Cross-validation weighted F1: 0.959 +/- 0.009
- Misclassified examples: 30
- Latest regenerated run: 96.24% accuracy and 96.24% weighted F1.
- Confidence analysis shows the model is usually decisive but still exposes a small uncertain slice.
- The misclassification report now captures concrete borderline examples for review.