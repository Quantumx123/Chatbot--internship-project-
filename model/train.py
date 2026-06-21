"""
Model Training Script for the Customer Service Chatbot.

This script:
  1. Loads the intents dataset (data/intents.json).
  2. Preprocesses all pattern texts using the NLP pipeline.
  3. Vectorizes texts with TF-IDF.
  4. Trains and compares multiple classifiers (LinearSVC, LogisticRegression, MultinomialNB).
  5. Selects the best model via stratified cross-validation.
  6. Saves the trained model and vectorizer for deployment.
  7. Prints a detailed classification report and confusion matrix.
"""

import json
import os
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
import joblib

# Add project root to path so we can import the nlp package.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from nlp.preprocessing import TextPreprocessor  # noqa: E402


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "intents.json")
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "trained_model.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")


# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
def load_intents(path: str) -> dict:
    """Load and return the intents JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 2. Prepare training data
# ---------------------------------------------------------------------------
def prepare_data(intents: dict, preprocessor: TextPreprocessor):
    """
    Extract (preprocessed_pattern, tag) pairs from the intents dataset.

    Returns:
        texts  – list of preprocessed pattern strings
        labels – list of corresponding intent tags
    """
    texts, labels = [], []
    for intent in intents["intents"]:
        tag = intent["tag"]
        for pattern in intent["patterns"]:
            processed = preprocessor.preprocess(pattern)
            texts.append(processed)
            labels.append(tag)
    return texts, labels


# ---------------------------------------------------------------------------
# 3. Train & evaluate
# ---------------------------------------------------------------------------
def train_and_evaluate(texts: list[str], labels: list[str]):
    """
    Vectorize the data, train multiple classifiers, compare with
    cross-validation, and return the best model + vectorizer.
    """
    # --- TF-IDF vectorization ---
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),   # unigrams + bigrams
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)

    # --- Classifiers to compare ---
    classifiers = {
        "LinearSVC": CalibratedClassifierCV(LinearSVC(max_iter=10000, C=1.0), cv=3),
        "LogisticRegression": LogisticRegression(max_iter=10000, C=1.0, solver="lbfgs"),
        "MultinomialNB": MultinomialNB(alpha=0.1),
    }

    # --- Cross-validation ---
    print("=" * 60)
    print("MODEL COMPARISON  (5-fold Stratified Cross-Validation)")
    print("=" * 60)

    best_name, best_score, best_clf = None, 0.0, None
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        mean_score = scores.mean()
        std_score = scores.std()
        print(f"  {name:<25s}  Accuracy: {mean_score:.4f} (+/- {std_score:.4f})")
        if mean_score > best_score:
            best_name, best_score, best_clf = name, mean_score, clf

    print(f"\n>>> Best model: {best_name} (accuracy: {best_score:.4f})")

    # --- Train best model on full data ---
    print(f"\nTraining {best_name} on the full dataset …")
    best_clf.fit(X, y)

    # --- Classification report on training data (indicative, not overfit
    #     because we already validated above with CV) ---
    y_pred = best_clf.predict(X)
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT  (full training data)")
    print("=" * 60)
    print(classification_report(y, y_pred))

    # --- Confusion matrix ---
    unique_labels = sorted(set(y))
    cm = confusion_matrix(y, y_pred, labels=unique_labels)
    print("CONFUSION MATRIX")
    print("-" * 60)
    # Header
    print(f"{'':>20s}", "  ".join(f"{l[:6]:>6s}" for l in unique_labels))
    for i, row in enumerate(cm):
        print(f"{unique_labels[i]:>20s}", "  ".join(f"{v:>6d}" for v in row))
    print()

    return best_clf, vectorizer, best_name


# ---------------------------------------------------------------------------
# 4. Save artefacts
# ---------------------------------------------------------------------------
def save_model(model, vectorizer, model_path: str, vectorizer_path: str):
    """Persist model and vectorizer to disk."""
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Model saved to     : {model_path}")
    print(f"Vectorizer saved to: {vectorizer_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading intents dataset …")
    intents = load_intents(DATA_PATH)
    n_intents = len(intents["intents"])
    n_patterns = sum(len(i["patterns"]) for i in intents["intents"])
    print(f"  Found {n_intents} intents with {n_patterns} total patterns.\n")

    preprocessor = TextPreprocessor()
    texts, labels = prepare_data(intents, preprocessor)

    model, vectorizer, model_name = train_and_evaluate(texts, labels)

    save_model(model, vectorizer, MODEL_PATH, VECTORIZER_PATH)
    print("\n[OK] Training complete!")


if __name__ == "__main__":
    main()
