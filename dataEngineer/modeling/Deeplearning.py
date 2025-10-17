import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import json
from tqdm.keras import TqdmCallback # ADDED: Import for the tqdm progress bar

class BertTextClassifier:
    """
    A deep learning classifier for text data using a pre-trained BERT model
    from the Hugging Face Transformers library and TensorFlow.
    ## ADDED: Now supports continual fine-tuning (incremental training).
    """
    def __init__(self, model_name='bert-base-uncased', model_dir='./bert_model', max_length=128):
        """
        Initializes the BERT classifier.

        Args:
            model_name (str): The name of the pre-trained BERT model to use from Hugging Face.
            model_dir (str): Directory to save/load the trained model and its components.
            max_length (int): The maximum length of a sequence for tokenization.
        """
        self.model_name = model_name
        self.model_dir = model_dir
        self.max_length = max_length
        self.model = None
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.label_encoder = LabelEncoder()
        
        os.makedirs(self.model_dir, exist_ok=True)
        self._label_encoder_path = os.path.join(self.model_dir, 'label_encoder.classes_.npy')
        self._config_path = os.path.join(self.model_dir, 'config.json')


    def _tokenize_data(self, texts):
        """Tokenizes the text data according to the BERT model's requirements."""
        return self.tokenizer(
            text=texts.tolist(),
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors='tf',
            return_token_type_ids=False,
            return_attention_mask=True,
            verbose=True
        )

    def save(self):
        """Saves the trained model, tokenizer, and label encoder."""
        if self.model:
            self.model.save_pretrained(self.model_dir)
            self.tokenizer.save_pretrained(self.model_dir)
            np.save(self._label_encoder_path, self.label_encoder.classes_)
            config = {
                'model_name': self.model_name,
                'max_length': self.max_length,
                'num_labels': len(self.label_encoder.classes_)
            }
            with open(self._config_path, 'w') as f:
                json.dump(config, f)
            print(f"Model, tokenizer, and configuration saved successfully to {self.model_dir}/")
        else:
            print("Error: No model to save.")

    def load(self):
        """Loads a trained model, tokenizer, and label encoder from disk."""
        ## MODIFIED: Changed check to use the config path for better reliability.
        if not os.path.exists(self._config_path):
             print(f"Error: Model configuration not found in directory {self.model_dir}/")
             return False
        try:
            with open(self._config_path, 'r') as f:
                config = json.load(f)
            
            self.label_encoder.classes_ = np.load(self._label_encoder_path, allow_pickle=True)
            self.tokenizer = BertTokenizer.from_pretrained(self.model_dir)
            self.model = TFBertForSequenceClassification.from_pretrained(self.model_dir)
            self.max_length = config['max_length']
            
            print(f"Model loaded successfully from {self.model_dir}/")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    ## MODIFIED: Major changes to support incremental training.
    def train(self, df, text_column, label_column, learning_rate=2e-5, epochs=3, batch_size=16, continuation_learning_rate=1e-5):
        """
        Trains or continues training the BERT classification model.
        If a model exists in model_dir, it will load it and continue training (fine-tuning).
        If not, it will train a new model from scratch.

        Args:
            df (pd.DataFrame): The dataframe with new training data.
            text_column (str): The name of the column with text data.
            label_column (str): The name of the column with labels.
            learning_rate (float): Learning rate for initial training.
            epochs (int): Number of training epochs.
            batch_size (int): The number of samples per batch.
            continuation_learning_rate (float): A smaller learning rate for fine-tuning an existing model.
        """
        # --- Check if a model already exists ---
        if os.path.exists(self._config_path):
            print("\n--- Found existing model. Starting continual fine-tuning. ---")
            self.load()
            current_lr = continuation_learning_rate
            
            # Verify that new data doesn't contain unseen labels
            print("Verifying labels...")
            new_labels = set(df[label_column].unique())
            known_labels = set(self.label_encoder.classes_)
            if not new_labels.issubset(known_labels):
                unknown = new_labels - known_labels
                raise ValueError(f"New data contains labels the model was not trained on: {unknown}. Incremental training requires the same set of labels.")
            
            # Use the loaded label encoder to transform new labels
            y = self.label_encoder.transform(df[label_column])
        else:
            print("\n--- No model found. Starting training from scratch. ---")
            current_lr = learning_rate
            # Fit the label encoder for the first time
            y = self.label_encoder.fit_transform(df[label_column])

        # --- Common data preparation steps ---
        X = df[text_column]
        num_labels = len(self.label_encoder.classes_)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y
        )
        print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples.")

        print("Tokenizing data...")
        X_train_encoded = self._tokenize_data(X_train)
        X_val_encoded = self._tokenize_data(X_val)

        # --- Model Initialization and Compilation ---
        if self.model is None: # If we are training from scratch
            print("Building new model...")
            self.model = TFBertForSequenceClassification.from_pretrained(
                self.model_name, num_labels=num_labels
            )

        optimizer = tf.keras.optimizers.Adam(learning_rate=current_lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
        
        # --- Train the model ---
        print(f"Training for {epochs} epochs with learning rate {current_lr}...")
        history = self.model.fit(
            x={'input_ids': X_train_encoded['input_ids'], 'attention_mask': X_train_encoded['attention_mask']},
            y=y_train,
            validation_data=(
                {'input_ids': X_val_encoded['input_ids'], 'attention_mask': X_val_encoded['attention_mask']},
                y_val
            ),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,  # ADDED: Turn off the default Keras progress bar
            callbacks=[TqdmCallback(verbose=1)]  # ADDED: Use the tqdm progress bar instead
        )
        
        print("\n✅ Training complete.")
        self.save()
        return history

    def predict(self, texts):
        """
        Predicts the labels for a list of new texts.

        Args:
            texts (list or pd.Series): A list of strings to classify.
        
        Returns:
            np.array: An array of predicted label strings.
        """
        if not self.model:
            if not self.load():
                print("Error: No model available. Please train or load a model first.")
                return None
        
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
            
        # Tokenize the input texts
        inputs = self._tokenize_data(pd.Series(texts))
        
        # Make predictions
        predictions = self.model.predict(
            {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']}
        )
        
        # The output of the model are logits, we need to find the class with the highest score
        predicted_class_ids = np.argmax(predictions.logits, axis=1)
        
        # Decode the predicted IDs back to original labels
        predicted_labels = self.label_encoder.inverse_transform(predicted_class_ids)
        
        return predicted_labels

    def evaluate(self, df, text_column, label_column, batch_size=32):
        """
        Evaluates the model on a test set.

        Args:
            df (pd.DataFrame): Dataframe with test data.
            text_column (str): Column with text.
            label_column (str): Column with true labels.
        """
        if not self.model:
            if not self.load():
                print("Error: No model available to evaluate.")
                return
        
        print("\n--- Evaluating model performance ---")
        X_test = df[text_column]
        y_test_labels = df[label_column]
        
        # Encode labels and tokenize text
        y_test = self.label_encoder.transform(y_test_labels)
        X_test_encoded = self._tokenize_data(X_test)
        
        # Evaluate
        loss, accuracy = self.model.evaluate(
            {'input_ids': X_test_encoded['input_ids'], 'attention_mask': X_test_encoded['attention_mask']},
            y_test,
            batch_size=batch_size
        )
        
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print(f"Test Loss: {loss:.4f}")


