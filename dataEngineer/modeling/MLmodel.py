import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import loguniform
import joblib
import os
import numpy as np

class TextClassifier:
    """
    A classifier for text data that supports incremental/batch learning and hyperparameter tuning.
    It can be trained from scratch or updated with new data without retraining.
    Uses SGDClassifier to function as either Logistic Regression or SVM.
    """
    def __init__(self, model_type='logreg', model_dir='/home/jax/NHA-112/models', use_hyperparameter_tuning=True):
        """
        Initializes the classifier.

        Args:
            model_type (str): 'logreg' for Logistic Regression or 'svm' for SVM.
            model_dir (str): Directory to save/load the model and vectorizer.
            use_hyperparameter_tuning (bool): If True, runs RandomizedSearchCV on first training.
        """
        if model_type not in ['logreg', 'svm']:
            raise ValueError("Unsupported model type. Choose 'logreg' or 'svm'.")
        
        self.model_type = model_type
        self.model_dir = model_dir
        self.vectorizer = None
        self.model = None
        self.use_hyperparameter_tuning = use_hyperparameter_tuning

        os.makedirs(self.model_dir, exist_ok=True)
        self._model_path = os.path.join(self.model_dir, f'{self.model_type}_sgd_model.pkl')
        self._vectorizer_path = os.path.join(self.model_dir, f'{self.model_type}_sgd_vectorizer.pkl')

    def _initialize_model(self):
        """Initializes a new SGDClassifier model based on the model_type."""
        if self.model_type == 'logreg':
            return SGDClassifier(loss='log_loss', random_state=42)
        elif self.model_type == 'svm':
            return SGDClassifier(loss='hinge', random_state=42)

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
            
    ## MODIFIED: Added the 'split_new_data_for_eval' parameter
    def train(self, df, text_column='processed_text', label_column='Rating', split_new_data_for_eval=True):
        """
        Trains the model.
        
        Args:
            df (pd.DataFrame): The dataframe with training data.
            text_column (str): Column with text data.
            label_column (str): Column with labels.
            eval_df (pd.DataFrame, optional): A dedicated dataframe for evaluation.
            split_new_data_for_eval (bool): If True and eval_df is None during an update,
                                            this will split 'df' into a temporary train/test
                                            set for before-and-after evaluation.
        """
        if os.path.exists(self._model_path):
            print("\n--- Found existing model. Performing incremental update. ---")
            self.load()
            
            # --- EVALUATION LOGIC ---
            # Priority 1: Use a dedicated evaluation dataframe if provided.
            
            
            ## NEW: Priority 2: Split the new data for evaluation if requested.
            if split_new_data_for_eval:
                print("\n`split_new_data_for_eval` is True. Splitting new data for evaluation...")
                if len(df) < 10:
                    print("Warning: New data batch is too small to split. Training on the full batch without evaluation.")
                    X_train_tfidf = self.vectorizer.transform(df[text_column])
                    self.model.partial_fit(X_train_tfidf, df[label_column])
                else:
                    # Split the new data batch into a temporary train and test set
                    train_batch_df, test_batch_df = train_test_split(
                        df, test_size=0.3, random_state=42, stratify=df[label_column]
                    )
                    
                    X_test_batch_tfidf = self.vectorizer.transform(test_batch_df[text_column])
                    y_test_batch = test_batch_df[label_column]
                    
                    # Evaluate OLD model on the temporary test set
                    self.evaluate(X_test_batch_tfidf, y_test_batch, title="--- Performance on New Batch BEFORE Update ---")
                    
                    # Train on the temporary training set
                    X_train_batch_tfidf = self.vectorizer.transform(train_batch_df[text_column])
                    self.model.partial_fit(X_train_batch_tfidf, train_batch_df[label_column])
                    
                    # Evaluate NEW model on the temporary test set
                    self.evaluate(X_test_batch_tfidf, y_test_batch, title="--- Performance on New Batch AFTER Update ---")
            
            # Priority 3: No evaluation, just train on the full batch.
            else:
                print("\nTraining on full new data batch without evaluation...")
                X_train_tfidf = self.vectorizer.transform(df[text_column])
                self.model.partial_fit(X_train_tfidf, df[label_column])
            
            print("\nModel update complete.")

        else:
            # This 'else' block for training from scratch remains the same
            print(f"\n--- No model found. Training new {self.model_type.upper()} model from scratch. ---")
            X = df[text_column]
            y = df[label_column]
            self.vectorizer = TfidfVectorizer(max_features=5000)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            X_train_tfidf = self.vectorizer.fit_transform(X_train)
            X_test_tfidf = self.vectorizer.transform(X_test)
            base_model = self._initialize_model()
            if self.use_hyperparameter_tuning:
                print("\n--- Performing Hyperparameter Tuning with RandomizedSearchCV ---")
                param_distributions = {
                    'alpha': loguniform(1e-5, 1e-1),
                    'penalty': ['l2', 'l1', 'elasticnet'],
                }
                if self.model_type == 'svm':
                    param_distributions['loss'] = ['hinge', 'modified_huber']
                random_search = RandomizedSearchCV(
                    base_model, param_distributions, n_iter=15, cv=3, verbose=1, random_state=42, n_jobs=-1
                )
                random_search.fit(X_train_tfidf, y_train)
                self.model = random_search.best_estimator_
                print(f"\nBest parameters found: {random_search.best_params_}")
            else:
                self.model = base_model
                all_possible_classes = np.unique(df[label_column])
                self.model.partial_fit(X_train_tfidf, y_train, classes=all_possible_classes)
            print("\nInitial model training complete.")
            self.evaluate(X_test_tfidf, y_test, title="--- Initial Model Performance on Test Set ---")

        self.save()

    def evaluate(self, X_test_tfidf, y_test, title="Model Evaluation"):
        """Evaluates the model on a test set."""
        print(f"\n{title}")
        y_pred = self.model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

    def predict(self, texts):
        """Predicts the rating for a list of new texts."""
        if not self.model and not self.load():
            print("Error: No model available. Please train the model first.")
            return None
        texts_tfidf = self.vectorizer.transform(texts)
        return self.model.predict(texts_tfidf)