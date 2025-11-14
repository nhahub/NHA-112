import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import loguniform
import joblib
import os
import numpy as np

class MultiTaskTextClassifier:
    """
    A multi-task classifier for text data.
    
    It trains a separate SGDClassifier for each specified label column,
    but shares a single TfidfVectorizer.
    Supports incremental/batch learning and hyperparameter tuning.

    NEW in V3:
    - No 'class_map' required.
    - On incremental updates, if a new class label is found,
      it prints a WARNING and skips that row for that task,
      preventing the 'ValueError' and allowing training to continue.
    """
    
    def __init__(self, label_columns: list, model_type='logreg', model_dir='/home/jax/NHA-112/models', use_hyperparameter_tuning=True):
        """
        Initializes the multi-task classifier.

        Args:
            label_columns (list): A list of string column names for each task.
                                  e.g., ['problem_type', 'category']
            model_type (str): 'logreg' for Logistic Regression or 'svm' for SVM.
            model_dir (str): Directory to save/load models and the vectorizer.
            use_hyperparameter_tuning (bool): If True, runs RandomizedSearchCV on first training.
        """
        if not label_columns:
            raise ValueError("`label_columns` list cannot be empty.")
        if model_type not in ['logreg', 'svm']:
            raise ValueError("Unsupported model type. Choose 'logreg' or 'svm'.")
            
        self.label_columns = label_columns
        self.model_type = model_type
        self.model_dir = model_dir
        self.use_hyperparameter_tuning = use_hyperparameter_tuning
        
        self.vectorizer = None
        self.models = {col: None for col in self.label_columns} # A dict to hold models

        os.makedirs(self.model_dir, exist_ok=True)
        
        # Paths for models and the shared vectorizer
        self._vectorizer_path = os.path.join(self.model_dir, f'multi_task_sgd_vectorizer.pkl')
        self._model_paths = {
            col: os.path.join(self.model_dir, f'{col}_{self.model_type}_sgd_model.pkl')
            for col in self.label_columns
        }

    def _initialize_model(self):
        """Initializes a new SGDClassifier model based on the model_type."""
        if self.model_type == 'logreg':
            return SGDClassifier(loss='log_loss', random_state=42)
        elif self.model_type == 'svm':
            return SGDClassifier(loss='hinge', random_state=42)

    def save(self):
        """Saves the trained models and shared vectorizer to disk."""
        if self.vectorizer:
            joblib.dump(self.vectorizer, self._vectorizer_path)
            print(f"Shared vectorizer saved to {self._vectorizer_path}")
        else:
            print("Error: No vectorizer to save.")
            return

        for col, model in self.models.items():
            if model:
                joblib.dump(model, self._model_paths[col])
                print(f"Model for task '{col}' saved to {self._model_paths[col]}")
            else:
                print(f"Error: No model for task '{col}' to save.")

    def load(self):
        """Loads all models and the vectorizer from disk."""
        try:
            self.vectorizer = joblib.load(self._vectorizer_path)
            for col in self.label_columns:
                self.models[col] = joblib.load(self._model_paths[col])
            
            print(f"All models and vectorizer loaded successfully from {self.model_dir}/")
            return True
        except FileNotFoundError:
            print("Could not find all required model/vectorizer files.")
            return False
            
    def train(self, df, text_column='processed_text', split_new_data_for_eval=True):
        """
        Trains all models, either from scratch or incrementally.
        """
        if os.path.exists(self._vectorizer_path):
            print("\n--- Found existing vectorizer. Performing incremental update for all tasks. ---")
            self.load()
            
            # --- MODIFIED: "SKIP AND WARN" LOGIC ---
            # This logic block handles the incremental update.
            
            # Note: 'split_new_data_for_eval' is complex with skipping.
            # For simplicity, we'll apply the skip logic to the whole batch,
            # then split if requested.
            
            print("\n--- Checking for new classes in update batch ---")
            df_to_train = df.copy()
            df_to_eval = None
            
            # We must filter the dataframe *before* transforming
            X_text = df[text_column]
            
            # This will hold the final boolean mask for rows that are
            # valid for *all* tasks (i.e., have no new labels)
            global_known_rows_mask = pd.Series(True, index=df.index)
            
            for col in self.label_columns:
                known_classes = set(self.models[col].classes_)
                
                # Get a boolean mask of rows that are in the known classes
                # We also treat 'None' or 'NaN' as skippable
                known_rows_mask = df[col].isin(known_classes) & df[col].notna()
                unknown_rows_mask = ~known_rows_mask
                
                if unknown_rows_mask.any():
                    num_unknown = unknown_rows_mask.sum()
                    new_examples = list(df.loc[unknown_rows_mask, col].unique())[:5]
                    print(f"  WARNING: Found {num_unknown} rows with new/unknown classes for task '{col}'.")
                    print(f"  New class examples: {new_examples}")
                    print(f"  These rows will be SKIPPED for this task's update.")
                    
                    # Update the global mask
                    global_known_rows_mask = global_known_rows_mask & known_rows_mask
            
            if global_known_rows_mask.all():
                print("  No new classes found in any task. Proceeding with full batch.")
                df_to_train = df
            elif not global_known_rows_mask.any():
                print("  CRITICAL: All rows in this batch contain new classes. No update can be performed.")
                return # Stop training
            else:
                num_skipped = (~global_known_rows_mask).sum()
                num_kept = global_known_rows_mask.sum()
                print(f"  Skipping {num_skipped} rows containing new classes.")
                print(f"  Proceeding with the remaining {num_kept} valid rows for training.")
                df_to_train = df[global_known_rows_mask]

            # Now, df_to_train contains *only* rows with known labels
            X_train_tfidf = self.vectorizer.transform(df_to_train[text_column])

            if split_new_data_for_eval and len(df_to_train) >= 10:
                print("\nSplitting filtered data for before/after evaluation...")
                # Split the *filtered* data
                try:
                    train_batch_df, test_batch_df = train_test_split(
                        df_to_train, test_size=0.3, random_state=42, stratify=df_to_train[self.label_columns[0]]
                    )
                except ValueError:
                    train_batch_df, test_batch_df = train_test_split(
                        df_to_train, test_size=0.3, random_state=42
                    )

                X_test_batch_tfidf = self.vectorizer.transform(test_batch_df[text_column])
                X_train_batch_tfidf = self.vectorizer.transform(train_batch_df[text_column])

                for col in self.label_columns:
                    y_test_batch = test_batch_df[col]
                    y_train_batch = train_batch_df[col]
                    
                    self.evaluate(self.models[col], X_test_batch_tfidf, y_test_batch, title=f"--- [{col}] Performance on New Batch BEFORE Update ---")
                    self.models[col].partial_fit(X_train_batch_tfidf, y_train_batch)
                    self.evaluate(self.models[col], X_test_batch_tfidf, y_test_batch, title=f"--- [{col}] Performance on New Batch AFTER Update ---")
            
            elif len(df_to_train) > 0:
                print("\nTraining on full filtered batch without evaluation...")
                for col in self.label_columns:
                    print(f"Updating model for task: '{col}'")
                    self.models[col].partial_fit(X_train_tfidf, df_to_train[col])
            else:
                print("\nNo valid data left to train on after filtering.")
            
            print("\nAll model updates complete.")

        else:
            # --- Train from scratch (This logic is now correct) ---
            print(f"\n--- No models found. Training new {self.model_type.upper()} models from scratch. ---")
            X = df[text_column]
            Y = df[self.label_columns]
            
            self.vectorizer = TfidfVectorizer(max_features=5000)
            
            try:
                X_train, X_test, Y_train, Y_test = train_test_split(
                    X, Y, test_size=0.2, random_state=42, stratify=Y[self.label_columns[0]]
                )
            except ValueError:
                print("Warning: Could not stratify. Splitting without stratification.")
                X_train, X_test, Y_train, Y_test = train_test_split(
                    X, Y, test_size=0.2, random_state=42
                )

            X_train_tfidf = self.vectorizer.fit_transform(X_train)
            X_test_tfidf = self.vectorizer.transform(X_test)
            
            # Loop and train a model for each task
            for col in self.label_columns:
                print(f"\n--- Training Initial Model for Task: '{col}' ---")
                y_train_task = Y_train[col]
                y_test_task = Y_test[col]
                
                # --- MODIFIED ---
                # Get all classes *from the entire initial dataframe*
                all_possible_classes = np.unique(df[col].dropna())
                print(f"Registering classes for '{col}': {all_possible_classes}")
                
                if self.use_hyperparameter_tuning:
                    print("\n--- Performing Hyperparameter Tuning with RandomizedSearchCV ---")
                    base_model = self._initialize_model()
                    param_distributions = {
                        'alpha': loguniform(1e-5, 1e-1),
                        'penalty': ['l2', 'l1', 'elasticnet'],
                    }
                    if self.model_type == 'svm':
                        param_distributions['loss'] = ['hinge', 'modified_huber']
                    
                    random_search = RandomizedSearchCV(
                        base_model, param_distributions, n_iter=15, cv=3, verbose=1, random_state=42, n_jobs=-1
                    )
                    random_search.fit(X_train_tfidf, y_train_task)
                    
                    print(f"\nBest parameters for '{col}': {random_search.best_params_}")

                    self.models[col] = self._initialize_model()
                    self.models[col].set_params(**random_search.best_params_)
                    
                    # First partial_fit registers all classes from the *full* initial DF
                    self.models[col].partial_fit(X_train_tfidf, y_train_task, classes=all_possible_classes)
                    
                else:
                    self.models[col] = self._initialize_model()
                    # First partial_fit registers all classes from the *full* initial DF
                    self.models[col].partial_fit(X_train_tfidf, y_train_task, classes=all_possible_classes)
                
                print(f"\nInitial model training complete for '{col}'.")
                self.evaluate(self.models[col], X_test_tfidf, y_test_task, title=f"--- Initial '{col}' Model Performance on Test Set ---")

        self.save()

    def evaluate(self, model, X_test_tfidf, y_test, title="Model Evaluation"):
        """Evaluates a single model on a test set."""
        print(f"\n{title}")
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        # Get all labels known to the model for a complete report
        all_labels = model.classes_
        print(classification_report(y_test, y_pred, labels=all_labels, zero_division=0))

    def predict(self, texts):
        """Predicts all tasks for a list of new texts."""
        if not self.vectorizer and not self.load():
            print("Error: No model available. Please train the model first.")
            return None
            
        texts_tfidf = self.vectorizer.transform(texts)
        
        predictions = {}
        for col, model in self.models.items():
            predictions[col] = model.predict(texts_tfidf)
            
        return predictions