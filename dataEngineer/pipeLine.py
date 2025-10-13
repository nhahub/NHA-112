import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    # Add this block to download the missing resource
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)


class NltkTextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # The input X is a pandas Series of texts.
        return X.apply(self._process_text)

    def _process_text(self, text):
        # a. Convert to lowercase and remove non-alphanumeric characters
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        # b. Tokenize
        tokens = word_tokenize(text)

        # c. Remove stopwords and lemmatize
        lemmatized_tokens = [
            self.lemmatizer.lemmatize(word)
            for word in tokens
            if word not in self.stop_words
        ]
        return " ".join(lemmatized_tokens)


def text_preprocessing_sklearn_pipeline(
    df: pd.DataFrame, text_col: str, rate_col: str
) -> pd.DataFrame:
    print(f"Initial dataset size: {len(df)} rows.")
    df.dropna(subset=[text_col, rate_col], inplace=True)
    df = df[df[text_col].str.strip() != ""]
    df.reset_index(drop=True, inplace=True)
    print(f"Size after removing empty rows: {len(df)} rows.")

    cleaning_pipeline = Pipeline([("text_preprocessor", NltkTextPreprocessor())])

    print("Applying sklearn pipeline for text cleaning and lemmatization...")
    processed_text_series = pd.Series(
        cleaning_pipeline.fit_transform(df[text_col]), name="processed_text"
    )
    df["processed_text"] = processed_text_series
    print("Text cleaning and lemmatization complete.")

    return df
