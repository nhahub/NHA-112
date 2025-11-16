import nltk, re, pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from deep_translator import GoogleTranslator  
from modeling.Deeplearning2 import MultiOutputClassificationModel

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


# -----------------------------
# Translator
# -----------------------------
def translate_arabic_to_english(text: str) -> str:
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception:
        return text  # fallback if translation fails


# load model
model_path = "/content/sentiment_classifier.pth"

model_handler = MultiOutputClassificationModel(
    model_name='distilbert-base-uncased',
    model_path=model_path
)

# Initialize preprocessor globally
preprocessor = NltkTextPreprocessor()

# -----------------------------
# Predict function
# -----------------------------
def predict(text: str):
    """
    Input: raw English or Arabic text
    Output: Predicted category & sub-category (dictionary)
    """
    # Step 1: Translate Arabic → English if needed
    try:
        text_en = GoogleTranslator(source='auto', target='en').translate(text)
    except Exception:
        text_en = text

    # Step 2: Preprocess using NLTK
    processed_text = preprocessor.transform(pd.Series([text_en]))[0]

    # Step 3: Predict using the BERT-based model
    prediction = model_handler.predict(processed_text)

    return prediction


'''
# -----------------------------
# Example Usage
# -----------------------------
# Predict on a sample text (any language)
text_arab = "المواصلات سيئة"
result = predict(text_arab)

print("Predicted Category:", result['category']['prediction'])
print("Predicted Sub-Category:", result['sub_category']['prediction'])
print("Top Category Predictions:", result['category']['top_predictions'])
print("Top Sub-Category Predictions:", result['sub_category']['top_predictions'])
'''

