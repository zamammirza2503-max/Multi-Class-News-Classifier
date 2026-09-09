"""
preprocess.py
-------------
Text cleaning utilities used before converting text into a
Bag of Words representation.

Steps performed:
    1. Lowercase the text
    2. Remove punctuation / digits / extra whitespace
    3. Remove stopwords (using scikit-learn's built-in English stopword list,
       so no external NLTK download is required)
    4. (Optional) Stem words using NLTK's PorterStemmer, which works fully
       offline since it is a pure algorithm and needs no downloaded corpus.
"""

import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

try:
    from nltk.stem import PorterStemmer
    _stemmer = PorterStemmer()
    _HAS_STEMMER = True
except ImportError:  # nltk not installed -> skip stemming gracefully
    _HAS_STEMMER = False


def clean_text(text: str, use_stemming: bool = True) -> str:
    """
    Clean a single piece of text for Bag-of-Words modeling.

    Parameters
    ----------
    text : str
        Raw input text (e.g. a news article).
    use_stemming : bool
        Whether to apply Porter stemming to each remaining token.

    Returns
    -------
    str
        Cleaned text, ready to be passed into CountVectorizer.
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # 3. Remove anything that isn't a letter or whitespace
    text = re.sub(r"[^a-z\s]", " ", text)

    # 4. Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    # 5. Tokenize (simple whitespace split)
    tokens = text.split()

    # 6. Remove stopwords and very short tokens
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) > 2]

    # 7. Optional stemming
    if use_stemming and _HAS_STEMMER:
        tokens = [_stemmer.stem(t) for t in tokens]

    return " ".join(tokens)


def clean_series(text_series, use_stemming: bool = True):
    """Apply clean_text() to a pandas Series of raw text."""
    return text_series.apply(lambda t: clean_text(t, use_stemming=use_stemming))


if __name__ == "__main__":
    sample = "The Company Reported RECORD Profits!!! Visit https://example.com for more info."
    print("Original :", sample)
    print("Cleaned  :", clean_text(sample))
