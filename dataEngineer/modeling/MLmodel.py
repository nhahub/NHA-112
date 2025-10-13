import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import numpy as np

class TextClassifier:
    """
    A classifier for text data that supports incremental/batch learning.
    It can be trained from scratch or updated with new data without retraining.
    Uses SGDClassifier to function as either Logistic Regression or SVM.
    """
    def __init__(self, model_type='logreg', model_dir='/home/jax/NHA-112/models'):
        """
        Initializes the classifier.

        Args:
            model_type (str): 'logreg' for Logistic Regression behavior, 'svm' for SVM.
            model_dir (str): Directory to save/load the model and vectorizer.
        """
        if model_type not in ['logreg', 'svm']:
            raise ValueError("Unsupported model type. Choose 'logreg' or 'svm'.")
        
        self.model_type = model_type
        self.model_dir = model_dir
        self.vectorizer = None
        self.model = None

        os.makedirs(self.model_dir, exist_ok=True)
        self._model_path = os.path.join(self.model_dir, f'{self.model_type}_sgd_model.pkl')
        self._vectorizer_path = os.path.join(self.model_dir, f'{self.model_type}_sgd_vectorizer.pkl')

    def _initialize_model(self):
        """Initializes a new SGDClassifier model based on the model_type."""
        if self.model_type == 'logreg':
            print("Initializing a new SGDClassifier with log_loss (Logistic Regression).")
            # loss='log_loss' makes SGDClassifier behave like Logistic Regression.
            self.model = SGDClassifier(loss='log_loss', random_state=42)
        elif self.model_type == 'svm':
            print("Initializing a new SGDClassifier with hinge loss (Linear SVM).")
            # loss='hinge' makes SGDClassifier behave like a linear SVM.
            self.model = SGDClassifier(loss='hinge', random_state=42)

    def save(self):
        """Saves the trained model and vectorizer to disk."""
        if self.model and self.vectorizer:
            joblib.dump(self.model, self._model_path)
            joblib.dump(self.vectorizer, self._vectorizer_path)
            print(f"Model and vectorizer saved successfully to {self.model_dir}/")
        else:
            print("Error: No model or vectorizer to save.")

    def load(self):
        """Loads a model and vectorizer from disk."""
        try:
            self.model = joblib.load(self._model_path)
            self.vectorizer = joblib.load(self._vectorizer_path)
            print(f"Model and vectorizer loaded successfully from {self.model_dir}/")
            return True
        except FileNotFoundError:
            return False

    def train(self, df, text_column='lemmatized_text', label_column='rate'):
        """
        Trains the model. Automatically detects if it should train from scratch
        or perform an incremental update based on whether a saved model exists.

        Args:
            df (pd.DataFrame): The dataframe with training data (can be full or a new batch).
            text_column (str): The name of the column with text data.
            label_column (str): The name of the column with labels.
        """
        X = df[text_column]
        y = df[label_column]

        # If a model exists, perform an incremental update (batch learning).
        if os.path.exists(self._model_path):
            print("\n--- Found existing model. Performing incremental update. ---")
            self.load()
            
            # Use the EXISTING vectorizer to transform the new data
            X_tfidf = self.vectorizer.transform(X)
            
            # Update the model with the new batch using partial_fit
            self.model.partial_fit(X_tfidf, y)
            print("Model update complete.")


        # If no model exists, train one from scratch.
        else:
            print(f"\n--- No model found. Training new {self.model_type.upper()} model from scratch. ---")
            self._initialize_model()
            self.vectorizer = TfidfVectorizer(max_features=1000)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Fit the vectorizer and transform the training data
            X_train_tfidf = self.vectorizer.fit_transform(X_train)
            X_test_tfidf = self.vectorizer.transform(X_test)
            
            # Use partial_fit for the first time, providing all possible class labels
            all_possible_classes = np.unique(df[label_column])
            self.model.partial_fit(X_train_tfidf, y_train, classes=all_possible_classes)
            print("Initial model training complete.")
            
            self.evaluate(X_test_tfidf, y_test)

        # Save the new or updated model
        self.save()

    def evaluate(self, X_test_tfidf, y_test):
        """Evaluates the model on a test set."""
        y_pred = self.model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

    def predict(self, texts):
        """Predicts the rating for a list of new texts."""
        if not self.load():
            print("Error: No model available. Please train the model first.")
            return None
        
        texts_tfidf = self.vectorizer.transform(texts)
        return self.model.predict(texts_tfidf)

