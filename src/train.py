"""
train.py
--------
End-to-end training pipeline for the Multi-Class News Classifier.

Pipeline:
    Raw CSV -> Text Cleaning -> Bag of Words (CountVectorizer)
            -> Train/Test Split -> Train several ML models
            -> Evaluate & pick the best one -> Save model + vectorizer

Run with:
    python src/train.py
"""

import os
import sys
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import clean_series  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "news_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            f"Run `python data/generate_dataset.py` first "
            f"(or point DATA_PATH to your own CSV with 'text' and 'category' columns)."
        )
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "category"]).drop_duplicates(subset=["text"])
    return df


def build_bag_of_words(train_texts, test_texts, max_features: int = 5000):
    """Fit a CountVectorizer (Bag of Words) on the training text only."""
    vectorizer = CountVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return vectorizer, X_train, X_test


def train_and_compare_models(X_train, y_train, X_test, y_test):
    """Train a handful of classic ML models and return the best one."""
    candidates = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(),
    }

    results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = {"model": model, "accuracy": acc, "preds": preds}
        print(f"{name:<28} -> accuracy: {acc:.4f}")

    best_name = max(results, key=lambda k: results[k]["accuracy"])
    print(f"\nBest model: {best_name} (accuracy = {results[best_name]['accuracy']:.4f})")
    return best_name, results


def save_confusion_matrix(y_test, preds, labels, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y_test, preds, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title("Confusion Matrix - Best Model")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix to: {out_path}")


def main():
    print("Loading data...")
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} rows across {df['category'].nunique()} categories.")
    print(df["category"].value_counts(), "\n")

    print("Cleaning text (lowercasing, removing stopwords/punctuation, stemming)...")
    df["clean_text"] = clean_series(df["text"])

    print("Splitting into train/test sets (80/20, stratified)...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean_text"], df["category"],
        test_size=0.2, random_state=42, stratify=df["category"]
    )

    print("Building Bag of Words features with CountVectorizer...")
    vectorizer, X_train, X_test = build_bag_of_words(X_train_text, X_test_text)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}\n")

    print("Training candidate models...")
    best_name, results = train_and_compare_models(X_train, y_train, X_test, y_test)
    best_model = results[best_name]["model"]
    best_preds = results[best_name]["preds"]

    print("\nClassification report for best model:")
    labels = sorted(df["category"].unique())
    print(classification_report(y_test, best_preds, labels=labels))

    # Save confusion matrix image
    save_confusion_matrix(y_test, best_preds, labels, os.path.join(DOCS_DIR, "confusion_matrix.png"))

    # Persist model + vectorizer + label list
    joblib.dump(best_model, os.path.join(MODELS_DIR, "news_classifier.pkl"))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "count_vectorizer.pkl"))
    joblib.dump(labels, os.path.join(MODELS_DIR, "labels.pkl"))

    with open(os.path.join(MODELS_DIR, "best_model_name.txt"), "w") as f:
        f.write(best_name)

    print(f"\nSaved trained model to:      {MODELS_DIR}/news_classifier.pkl")
    print(f"Saved vectorizer to:         {MODELS_DIR}/count_vectorizer.pkl")
    print(f"Saved label list to:         {MODELS_DIR}/labels.pkl")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
