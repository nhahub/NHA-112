import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class BertTextClassifier:
    """
    BERT text classifier with safe save/load and incremental learning (supports adding new labels).
    Usage: initialize with model_dir pointing to where you want model artifacts saved (e.g. on Drive).
    Methods: train(df, text_column, label_column, ...), predict(texts), evaluate(df,...), save(), load()
    """
    def __init__(self, model_name='bert-base-uncased', model_dir='./bert_model', max_length=128):
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
        if isinstance(texts, (list, np.ndarray)):
            s = pd.Series(texts)
        elif isinstance(texts, pd.Series):
            s = texts
        else:
            s = pd.Series(texts)
        return self.tokenizer(
            text=s.tolist(),
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors='tf',
            return_token_type_ids=False,
            return_attention_mask=True,
        )

    # Helper: map textual labels to indices using current label_encoder.classes_
    def _labels_to_indices(self, labels):
        classes = list(self.label_encoder.classes_)
        mapping = {c: i for i, c in enumerate(classes)}
        return np.array([mapping[l] for l in labels], dtype=np.int32)

    def save(self):
        """Save model, tokenizer and label encoder. Saves TensorFlow format by default."""
        if self.model is None:
            print("❌ No model to save.")
            return
        # Save TF format model (transformers will write tf_model.h5 / config.json)
        self.model.save_pretrained(self.model_dir)  # saves tf model files
        self.tokenizer.save_pretrained(self.model_dir)
        np.save(self._label_encoder_path, self.label_encoder.classes_)
        cfg = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'num_labels': len(self.label_encoder.classes_)
        }
        with open(self._config_path, 'w') as f:
            json.dump(cfg, f)
        print(f"✅ Saved model + tokenizer + label encoder to {self.model_dir}")

    def load(self):
        """Load model, tokenizer and label encoder from model_dir."""
        if not os.path.exists(self._config_path):
            print(f"⚠️ No config found in {self.model_dir}. Nothing to load.")
            return False

        try:
            with open(self._config_path, 'r') as f:
                cfg = json.load(f)
            self.max_length = cfg.get('max_length', self.max_length)

            # load label encoder classes
            if os.path.exists(self._label_encoder_path):
                self.label_encoder.classes_ = np.load(self._label_encoder_path, allow_pickle=True)
            else:
                # if missing, create empty classes (user will fit on training)
                self.label_encoder.classes_ = np.array([])

            # tokenizer
            self.tokenizer = BertTokenizer.from_pretrained(self.model_dir)

            # Detect whether directory contains PyTorch weights
            pytorch_bin = os.path.join(self.model_dir, "pytorch_model.bin")
            has_pt = os.path.exists(pytorch_bin)

            # Attempt TF load first if no PyTorch file present; otherwise load from_pt=True
            from_pt_flag = True if has_pt else False

            # If tf files exist, from_pt=False is better; but transformers handles both.
            try:
                self.model = TFBertForSequenceClassification.from_pretrained(self.model_dir, from_pt=from_pt_flag)
            except Exception as e:
                # fallback: try forcing the other option
                try:
                    self.model = TFBertForSequenceClassification.from_pretrained(self.model_dir, from_pt=not from_pt_flag)
                except Exception as e2:
                    raise RuntimeError(f"Failed to load model from {self.model_dir}: {e} | {e2}")

            print(f"✅ Loaded model from {self.model_dir} (from_pt={from_pt_flag})")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def train(self, df, text_column, label_column,
              learning_rate=2e-5, epochs=3, batch_size=16, continuation_learning_rate=1e-5):
        """
        Train or continue training. Supports adding new labels (incremental learning).
        If new labels appear, the head is expanded and BERT encoder weights are transferred.
        """
        # Prepare labels: check if we already have saved label classes
        existing_labels = set()
        if os.path.exists(self._config_path) and os.path.exists(self._label_encoder_path):
            try:
                existing_labels = set(np.load(self._label_encoder_path, allow_pickle=True).tolist())
            except Exception:
                existing_labels = set()

        new_labels_in_data = set(df[label_column].unique())
        unseen_labels = new_labels_in_data - existing_labels

        # If a saved model exists, load it
        if os.path.exists(self._config_path):
            loaded = self.load()
            if not loaded:
                # If load failed, remove inconsistent dir to avoid future confusion and start fresh
                print("⚠️ Failed to load existing model. Training from scratch.")
                self.model = None

        # If unseen labels exist and we had a previous label set, we will extend
        if len(existing_labels) > 0 and len(unseen_labels) > 0:
            print(f"--- Detected new labels during incremental training: {unseen_labels} ---")
            # Build new classes array (keep previous order, then append new labels in sorted order)
            old_classes = list(np.load(self._label_encoder_path, allow_pickle=True))
            # Append new labels preserving user-supplied order
            appended = [l for l in df[label_column].unique() if l not in old_classes]
            new_classes = old_classes + appended
            self.label_encoder.classes_ = np.array(new_classes, dtype=object)
            new_num_labels = len(new_classes)

            # Build a new model with new_num_labels and transfer encoder weights from the old model if available
            print("Building new model with expanded head and transferring encoder weights (if available)...")
            new_model = TFBertForSequenceClassification.from_pretrained(self.model_name, num_labels=new_num_labels, from_pt=False)

            # if old model exists, transfer encoder/bert weights
            if self.model is not None:
                try:
                    # Transfer the BERT encoder weights (the base model)
                    new_model.bert.set_weights(self.model.bert.get_weights())
                    print("✅ Copied BERT encoder weights to the new model.")
                except Exception as e:
                    print(f"⚠️ Failed to copy encoder weights: {e} — continuing with fresh encoder.")

            # replace model with new one (classifier head is new & randomly initialized)
            self.model = new_model

        else:
            # No unseen labels or first-time training: fit/refresh label encoder normally
            if len(existing_labels) == 0:
                # Fit label encoder for the first time
                self.label_encoder.fit(df[label_column])
            else:
                # keep existing classes (no changes)
                if len(self.label_encoder.classes_) == 0:
                    self.label_encoder.classes_ = np.load(self._label_encoder_path, allow_pickle=True)

        # Now we have self.label_encoder.classes_ set to the desired classes
        num_labels = len(self.label_encoder.classes_)
        if num_labels == 0:
            raise ValueError("No labels found to train on.")

        # Tokenize and split
        X = df[text_column]
        y = self._labels_to_indices(df[label_column].tolist())

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y
        )
        print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples.")

        X_train_encoded = self._tokenize_data(X_train)
        X_val_encoded = self._tokenize_data(X_val)

        # If we don't have a model instance yet, instantiate one with correct num_labels
        if self.model is None:
            print("Building model from pretrained transformer (TensorFlow weights).")
            self.model = TFBertForSequenceClassification.from_pretrained(self.model_name, num_labels=num_labels, from_pt=False)

        # If model exists but its classification head size differs from current num_labels, rebuild model and transfer encoder
        try:
            current_head_out = self.model.classifier.out_features if hasattr(self.model, 'classifier') and hasattr(self.model.classifier, 'out_features') else None
        except Exception:
            current_head_out = None

        # Compile
        current_lr = continuation_learning_rate if os.path.exists(self._config_path) else learning_rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=current_lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

        # Fit
        print(f"Training for {epochs} epochs with lr={current_lr} ...")
        history = self.model.fit(
            x={'input_ids': X_train_encoded['input_ids'], 'attention_mask': X_train_encoded['attention_mask']},
            y=y_train,
            validation_data=(
                {'input_ids': X_val_encoded['input_ids'], 'attention_mask': X_val_encoded['attention_mask']},
                y_val
            ),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        # Save updated model and label encoder/classes
        # Ensure label encoder classes file is written before saving model config
        np.save(self._label_encoder_path, self.label_encoder.classes_)
        cfg = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'num_labels': len(self.label_encoder.classes_)
        }
        with open(self._config_path, 'w') as f:
            json.dump(cfg, f)

        self.save()  # save model + tokenizer + label encoder
        return history

    def predict(self, texts):
        """Return predicted label strings for input texts (list or pd.Series)."""
        if self.model is None:
            if not self.load():
                raise RuntimeError("No model available. Train or save a model first.")
        if isinstance(texts, str):
            texts = [texts]
        inputs = self._tokenize_data(pd.Series(texts))
        outputs = self.model.predict({'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']})
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        preds = np.argmax(logits, axis=1)
        # map indices back to labels
        classes = list(self.label_encoder.classes_)
        return np.array([classes[i] for i in preds])

    def evaluate(self, df, text_column, label_column, batch_size=32):
        """Evaluate model on a dataframe and print accuracy & loss."""
        if self.model is None:
            if not self.load():
                raise RuntimeError("No model available to evaluate.")
        X_test = df[text_column]
        y_test = self._labels_to_indices(df[label_column].tolist())
        X_test_encoded = self._tokenize_data(X_test)
        loss, acc = self.model.evaluate(
            {'input_ids': X_test_encoded['input_ids'], 'attention_mask': X_test_encoded['attention_mask']},
            y_test,
            batch_size=batch_size,
            verbose=1
        )
        print(f"\nTest Loss: {loss:.4f}  Test Accuracy: {acc:.4f}")
        return loss, acc
