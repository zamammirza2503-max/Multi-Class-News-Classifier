"""
app.py
------
A small Streamlit web app that lets you paste in a news headline / snippet
and see the predicted category live.

Run with:
    streamlit run app.py

Make sure you've trained the model first:
    python data/generate_dataset.py
    python src/train.py
"""

import os
import sys
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.predict import load_artifacts, predict_with_confidence  # noqa: E402

st.set_page_config(page_title="News Classifier", layout="centered")

st.title(" Multi-Class News Classifier")
st.write(
    "Paste a news headline or short article below and this app will predict "
    "which category it belongs to, using a **Bag of Words + Machine Learning** model."
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


@st.cache_resource
def get_model():
    return load_artifacts()


try:
    model, vectorizer = get_model()
except FileNotFoundError:
    st.error(
        "No trained model found. Please run the following in your terminal first:\n\n"
        "```bash\npython data/generate_dataset.py\npython src/train.py\n```"
    )
    st.stop()

text_input = st.text_area(
    "News text",
    placeholder="e.g. The central bank raised interest rates for the third time this year...",
    height=150,
)

if st.button("Classify", type="primary"):
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        prediction, probs = predict_with_confidence(text_input, model, vectorizer)
        st.success(f"Predicted category: **{prediction}**")

        if probs:
            st.subheader("Confidence per category")
            sorted_probs = dict(sorted(probs.items(), key=lambda x: -x[1]))
            st.bar_chart(sorted_probs)

st.divider()
st.caption(
    "Model: Bag of Words (CountVectorizer) + classic ML classifier "
    "(Naive Bayes / Logistic Regression / Linear SVM — best one is auto-selected during training)."
)
