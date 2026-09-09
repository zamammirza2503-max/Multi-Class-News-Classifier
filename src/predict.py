"""
predict.py
----------
Load the trained Bag-of-Words model and classify new news text
from the command line.

Usage:
    python src/predict.py "Apple unveiled its new AI-powered chip today."
    python src/predict.py    # interactive mode, prompts for input
"""

import os
import sys
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import clean_text  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_artifacts():
    model_path = os.path.join(MODELS_DIR, "news_classifier.pkl")
    vec_path = os.path.join(MODELS_DIR, "count_vectorizer.pkl")

    if not (os.path.exists(model_path) and os.path.exists(vec_path)):
        raise FileNotFoundError(
            "Trained model not found. Run `python src/train.py` first."
        )

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer


def predict_category(text: str, model, vectorizer) -> str:
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    return prediction


def predict_with_confidence(text: str, model, vectorizer):
    """Return (prediction, {category: probability}) if the model supports it."""
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        prob_dict = dict(zip(model.classes_, probs))
        return prediction, prob_dict
    return prediction, None


def main():
    model, vectorizer = load_artifacts()

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = input("Enter a news headline / article snippet: ")

    prediction, probs = predict_with_confidence(text, model, vectorizer)

    print(f"\nInput text : {text}")
    print(f"Predicted category: {prediction}")

    if probs:
        print("\nConfidence per category:")
        for category, prob in sorted(probs.items(), key=lambda x: -x[1]):
            print(f"  {category:<15} {prob * 100:5.1f}%")


if __name__ == "__main__":
    main()
