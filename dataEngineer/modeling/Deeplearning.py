import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
from google.colab import drive # Import the drive module

# ---------------- SENTIMENT CLASSIFIER MODEL ----------------
class SentimentClassifier(nn.Module):
    """The neural network architecture for sentiment classification."""
    def __init__(self, model_name='bert-base-uncased', num_classes=3, dropout_rate=0.3):
        super(SentimentClassifier, self).__init__()
        self.pretrained_model = AutoModel.from_pretrained(model_name)

        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        
        if hasattr(self.pretrained_model, 'encoder'):
            layers_to_unfreeze = self.pretrained_model.encoder.layer[-4:]
        elif hasattr(self.pretrained_model, 'transformer'):
            layers_to_unfreeze = self.pretrained_model.transformer.layer[-4:]
        else:
            print("Warning: Could not identify layers to unfreeze.")
            layers_to_unfreeze = []

        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(self.pretrained_model.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# ---------------- PYTORCH DATASET ----------------
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text, truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ---------------- MAIN MODEL HANDLER CLASS ----------------
class SentimentAnalysisModel:
    def __init__(self, model_name='bert-base-uncased', model_path='sentiment_classifier.pth'):
        self.model_name = model_name
        self.model_path = model_path
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label_encoder = LabelEncoder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def save(self, epoch, optimizer, loss, accuracy):
        if self.model:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss, 'accuracy': accuracy,
                'label_encoder_classes': self.label_encoder.classes_,
                'model_name': self.model_name,
                'num_classes': len(self.label_encoder.classes_)
            }, self.model_path)
            print(f"✅ Model saved to {self.model_path} with accuracy: {accuracy:.2f}%")
        else:
            print("Error: No model to save.")

    def load(self):
        if not os.path.exists(self.model_path):
            print(f"Error: Model file not found at {self.model_path}")
            return None, None
        # FIX: Added weights_only=False to allow loading the LabelEncoder object
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        num_classes = checkpoint['num_classes']
        model_name = checkpoint.get('model_name', self.model_name)
        self.model = SentimentClassifier(model_name=model_name, num_classes=num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.label_encoder.classes_ = checkpoint['label_encoder_classes']
        optimizer = optim.AdamW(self.model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✅ Loaded model from {self.model_path}.")
        return self.model, optimizer

    def train(self, df, text_column, label_column, num_epochs=5, batch_size=16, 
              learning_rate=2e-5, continuation_learning_rate=1e-5):
        optimizer = None
        if os.path.exists(self.model_path):
            print("\n--- Found existing model. Starting incremental training. ---")
            self.model, optimizer = self.load()
            current_lr = continuation_learning_rate
            encoded_labels = self.label_encoder.transform(df[label_column])
        else:
            print("\n--- No model found. Starting training from scratch. ---")
            encoded_labels = self.label_encoder.fit_transform(df[label_column])
            num_classes = len(self.label_encoder.classes_)
            self.model = SentimentClassifier(model_name=self.model_name, num_classes=num_classes)
            self.model.to(self.device)
            current_lr = learning_rate

        texts = df[text_column].values
        print("Categories:", list(self.label_encoder.classes_))

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, encoded_labels, test_size=0.2, random_state=42
        )
        train_dataset = SentimentDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = SentimentDataset(val_texts, val_labels, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        if optimizer is None:
            optimizer = optim.AdamW(self.model.parameters(), lr=current_lr)
        
        for g in optimizer.param_groups:
            g['lr'] = current_lr
        criterion = nn.CrossEntropyLoss()
        
        print(f"Starting training for {num_epochs} epochs with learning rate {current_lr}...")
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            self.model.eval()
            correct, total, val_loss = 0, 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    outputs = self.model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total if total > 0 else 0
            avg_train_loss = total_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0

            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Accuracy: {accuracy:.2f}%")
        
        self.save(epoch, optimizer, avg_val_loss, accuracy)
        print("✅ Training session complete!")

    # --- NEWLY ADDED PREDICTION METHOD ---
    def predict(self, text):
        """Predicts the sentiment for a single piece of text."""
        if not self.model:
            print("Model not in memory. Attempting to load from path...")
            self.load()
            if not self.model:
                print("Error: Could not load model. Please train a model first.")
                return None, None, None

        self.model.eval() # Set the model to evaluation mode

        # Tokenize the input text
        encoding = self.tokenizer(text, truncation=True, padding='max_length',
                                  max_length=512, return_tensors='pt')
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Get model predictions
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class_id = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class_id].item()

        # Decode the prediction
        predicted_sentiment = self.label_encoder.inverse_transform([predicted_class_id])[0]

        # Get top predictions
        top_probs, top_indices = torch.topk(probabilities[0], k=min(3, len(self.label_encoder.classes_)))
        sentiments = self.label_encoder.inverse_transform(top_indices.cpu().numpy())
        confidences = top_probs.cpu().numpy()
        top_predictions = {s: float(c) for s, c in zip(sentiments, confidences)}

        return predicted_sentiment, confidence, top_predictions

