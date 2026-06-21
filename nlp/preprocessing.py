"""
Text Preprocessing Pipeline for the Customer Service Chatbot.

This module handles all text cleaning, tokenization, lemmatization,
and vectorization tasks required before model training and inference.
"""

import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ---------------------------------------------------------------------------
# Ensure required NLTK resources are available
# ---------------------------------------------------------------------------
def download_nltk_data():
    """Download required NLTK datasets if not already present."""
    resources = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]
    for resource in resources:
        try:
            nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else resource)
        except LookupError:
            nltk.download(resource, quiet=True)


download_nltk_data()


# ---------------------------------------------------------------------------
# Common contractions mapping
# ---------------------------------------------------------------------------
CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "i'm": "i am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "they're": "they are",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "we'll": "we will",
    "they'll": "they will",
    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would",
    "what's": "what is",
    "that's": "that is",
    "there's": "there is",
    "here's": "here is",
    "where's": "where is",
    "who's": "who is",
    "how's": "how is",
    "let's": "let us",
}

# Words to KEEP even though they are in the default stop-word list,
# because they carry intent-level meaning for a customer-service chatbot.
KEEP_STOPWORDS = {"not", "no", "nor", "don", "do", "does", "did", "can",
                  "will", "won", "should", "could", "would", "how", "when",
                  "where", "what", "why", "my", "your", "i"}


# ---------------------------------------------------------------------------
# Preprocessing class
# ---------------------------------------------------------------------------
class TextPreprocessor:
    """End-to-end text preprocessor: clean → tokenize → lemmatize → rejoin."""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        # Build a filtered stop-word set (remove words we want to keep).
        self.stop_words = set(stopwords.words("english")) - KEEP_STOPWORDS

    # ----- individual steps ------------------------------------------------

    def expand_contractions(self, text: str) -> str:
        """Replace common English contractions with their expanded forms."""
        for contraction, expansion in CONTRACTIONS.items():
            text = text.replace(contraction, expansion)
        return text

    def clean_text(self, text: str) -> str:
        """Lowercase, expand contractions, strip special chars."""
        text = text.lower().strip()
        text = self.expand_contractions(text)
        # Remove anything that is not a letter, digit, or whitespace.
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        # Collapse multiple whitespace.
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> list[str]:
        """Tokenize with NLTK's word_tokenize."""
        return word_tokenize(text)

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        """Remove stop words while keeping intent-relevant ones."""
        return [t for t in tokens if t not in self.stop_words]

    def lemmatize(self, tokens: list[str]) -> list[str]:
        """Lemmatize each token to its base form."""
        return [self.lemmatizer.lemmatize(t) for t in tokens]

    # ----- full pipeline ---------------------------------------------------

    def preprocess(self, text: str) -> str:
        """
        Run the full preprocessing pipeline and return the cleaned text
        as a single string (ready for TF-IDF vectorization).
        """
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return " ".join(tokens)
