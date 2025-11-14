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
# from google.colab import drive # Import the drive module

# ---------------- MULTI-OUTPUT CLASSIFIER MODEL ----------------
class MultiOutputClassifier(nn.Module):
    """
    The neural network architecture for multi-output classification.
    It has two separate classification heads, one for 'category' and one for 'sub_category'.
    """
    def __init__(self, model_name='bert-base-uncased', num_classes_category=10, num_classes_subcategory=4, dropout_rate=0.3):
        super(MultiOutputClassifier, self).__init__()
        self.pretrained_model = AutoModel.from_pretrained(model_name)

        # --- Freeze/Unfreeze Layers (same as original) ---
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
        # --- End Freeze/Unfreeze ---

        self.dropout = nn.Dropout(dropout_rate)
        
        # --- MODIFICATION: Two separate classifier heads ---
        shared_hidden_size = self.pretrained_model.config.hidden_size
        
        # Head 1: For main category
        self.classifier_category = nn.Sequential(
            nn.Linear(shared_hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes_category) # Output size for category
        )
        
        # Head 2: For sub-category
        self.classifier_subcategory = nn.Sequential(
            nn.Linear(shared_hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes_subcategory) # Output size for sub-category
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :] # Use [CLS] token
        pooled_output = self.dropout(pooled_output)
        
        # --- MODIFICATION: Pass pooled output to both heads ---
        logits_category = self.classifier_category(pooled_output)
        logits_subcategory = self.classifier_subcategory(pooled_output)
        
        return logits_category, logits_subcategory # Return both sets of logits

# ---------------- PYTORCH DATASET ----------------
class MultiOutputDataset(Dataset):
    """
    Dataset class modified to handle two label outputs.
    """
    def __init__(self, texts, labels_category, labels_subcategory, tokenizer, max_length=512):
        self.texts = texts
        self.labels_category = labels_category # Labels for the first task
        self.labels_subcategory = labels_subcategory # Labels for the second task
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label_cat = self.labels_category[idx]
        label_subcat = self.labels_subcategory[idx]
        
        encoding = self.tokenizer(
            text, truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels_category': torch.tensor(label_cat, dtype=torch.long), # Return category label
            'labels_subcategory': torch.tensor(label_subcat, dtype=torch.long) # Return sub-category label
        }

# ---------------- MAIN MODEL HANDLER CLASS ----------------
class MultiOutputClassificationModel:
    def __init__(self, model_name='bert-base-uncased', model_path='multi_output_classifier.pth'):
        self.model_name = model_name
        self.model_path = model_path
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # --- MODIFICATION: Two Label Encoders ---
        self.label_encoder_category = LabelEncoder()
        self.label_encoder_subcategory = LabelEncoder()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def save(self, epoch, optimizer, loss, accuracy_cat, accuracy_subcat):
        if self.model:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'accuracy_category': accuracy_cat,
                'accuracy_subcategory': accuracy_subcat,
                
                # --- MODIFICATION: Save classes for both encoders ---
                'label_encoder_category_classes': self.label_encoder_category.classes_,
                'label_encoder_subcategory_classes': self.label_encoder_subcategory.classes_,
                'num_classes_category': len(self.label_encoder_category.classes_),
                'num_classes_subcategory': len(self.label_encoder_subcategory.classes_),
                
                'model_name': self.model_name,
            }, self.model_path)
            print(f"✅ Model saved to {self.model_path} with Acc (Cat): {accuracy_cat:.2f}%, Acc (SubCat): {accuracy_subcat:.2f}%")
        else:
            print("Error: No model to save.")

    def load(self):
        if not os.path.exists(self.model_path):
            print(f"Error: Model file not found at {self.model_path}")
            return None, None
            
        # FIX: Added weights_only=False to allow loading the LabelEncoder object
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False) 
        
        # --- MODIFICATION: Load classes for both encoders ---
        num_classes_category = checkpoint['num_classes_category']
        num_classes_subcategory = checkpoint['num_classes_subcategory']
        model_name = checkpoint.get('model_name', self.model_name)
        
        self.model = MultiOutputClassifier(
            model_name=model_name, 
            num_classes_category=num_classes_category, 
            num_classes_subcategory=num_classes_subcategory
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
        self.label_encoder_category.classes_ = checkpoint['label_encoder_category_classes']
        self.label_encoder_subcategory.classes_ = checkpoint['label_encoder_subcategory_classes']
        
        optimizer = optim.AdamW(self.model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"✅ Loaded model from {self.model_path}.")
        return self.model, optimizer

    def train(self, df, text_column, category_column, subcategory_column, num_epochs=5, batch_size=16, 
              learning_rate=2e-5, continuation_learning_rate=1e-5):
        
        optimizer = None
        if os.path.exists(self.model_path):
            print("\n--- Found existing model. Starting incremental training. ---")
            self.model, optimizer = self.load()
            current_lr = continuation_learning_rate
            # --- MODIFICATION: Transform both label columns ---
            encoded_labels_cat = self.label_encoder_category.transform(df[category_column])
            encoded_labels_subcat = self.label_encoder_subcategory.transform(df[subcategory_column])
        else:
            print("\n--- No model found. Starting training from scratch. ---")
            # --- MODIFICATION: Fit_transform both label columns ---
            encoded_labels_cat = self.label_encoder_category.fit_transform(df[category_column])
            encoded_labels_subcat = self.label_encoder_subcategory.fit_transform(df[subcategory_column])
            
            num_classes_cat = len(self.label_encoder_category.classes_)
            num_classes_subcat = len(self.label_encoder_subcategory.classes_)
            
            self.model = MultiOutputClassifier(
                model_name=self.model_name, 
                num_classes_category=num_classes_cat, 
                num_classes_subcategory=num_classes_subcat
            )
            self.model.to(self.device)
            current_lr = learning_rate

        texts = df[text_column].values
        print("Categories:", list(self.label_encoder_category.classes_))
        print("Sub-Categories:", list(self.label_encoder_subcategory.classes_))

        train_texts, val_texts, train_labels_cat, val_labels_cat, train_labels_subcat, val_labels_subcat = train_test_split(
            texts, encoded_labels_cat, encoded_labels_subcat, test_size=0.2, random_state=42
        )
        
        train_dataset = MultiOutputDataset(train_texts, train_labels_cat, train_labels_subcat, self.tokenizer)
        val_dataset = MultiOutputDataset(val_texts, val_labels_cat, val_labels_subcat, self.tokenizer)
        
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
                labels_cat = batch['labels_category'].to(self.device)
                labels_subcat = batch['labels_subcategory'].to(self.device)
                
                outputs_cat, outputs_subcat = self.model(input_ids, attention_mask)
                
                # --- MODIFICATION: Calculate two losses and combine them ---
                loss_cat = criterion(outputs_cat, labels_cat)
                loss_subcat = criterion(outputs_subcat, labels_subcat)
                loss = loss_cat + loss_subcat # Simple sum, could be weighted
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            self.model.eval()
            # --- MODIFICATION: Track accuracy for both tasks ---
            correct_cat, correct_subcat, total, val_loss = 0, 0, 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels_cat = batch['labels_category'].to(self.device)
                    labels_subcat = batch['labels_subcategory'].to(self.device)
                    
                    outputs_cat, outputs_subcat = self.model(input_ids, attention_mask)
                    
                    loss_cat = criterion(outputs_cat, labels_cat)
                    loss_subcat = criterion(outputs_subcat, labels_subcat)
                    loss = loss_cat + loss_subcat
                    val_loss += loss.item()
                    
                    _, predicted_cat = torch.max(outputs_cat.data, 1)
                    _, predicted_subcat = torch.max(outputs_subcat.data, 1)
                    
                    total += labels_cat.size(0) # Total is the same for both
                    correct_cat += (predicted_cat == labels_cat).sum().item()
                    correct_subcat += (predicted_subcat == labels_subcat).sum().item()

            accuracy_cat = 100 * correct_cat / total if total > 0 else 0
            accuracy_subcat = 100 * correct_subcat / total if total > 0 else 0
            avg_train_loss = total_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0

            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Acc (Cat): {accuracy_cat:.2f}% | "
                  f"Acc (SubCat): {accuracy_subcat:.2f}%")
        
        self.save(epoch, optimizer, avg_val_loss, accuracy_cat, accuracy_subcat)
        print("✅ Training session complete!")

    def predict(self, text):
        """Predicts the category and sub-category for a single piece of text."""
        if not self.model:
            print("Model not in memory. Attempting to load from path...")
            self.load()
            if not self.model:
                print("Error: Could not load model. Please train a model first.")
                return None, None

        self.model.eval() # Set the model to evaluation mode

        encoding = self.tokenizer(text, truncation=True, padding='max_length',
                                  max_length=512, return_tensors='pt')
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # --- MODIFICATION: Get predictions from both heads ---
        with torch.no_grad():
            outputs_cat, outputs_subcat = self.model(input_ids, attention_mask)
            
            # Process Category Prediction
            probs_cat = torch.softmax(outputs_cat, dim=1)
            pred_class_id_cat = torch.argmax(probs_cat, dim=1).item()
            confidence_cat = probs_cat[0][pred_class_id_cat].item()
            predicted_category = self.label_encoder_category.inverse_transform([pred_class_id_cat])[0]
            
            # Process Sub-Category Prediction
            probs_subcat = torch.softmax(outputs_subcat, dim=1)
            pred_class_id_subcat = torch.argmax(probs_subcat, dim=1).item()
            confidence_subcat = probs_subcat[0][pred_class_id_subcat].item()
            predicted_subcategory = self.label_encoder_subcategory.inverse_transform([pred_class_id_subcat])[0]

        # --- Get top predictions for both ---
        
        # Top Category Predictions
        top_probs_cat, top_indices_cat = torch.topk(probs_cat[0], k=min(3, len(self.label_encoder_category.classes_)))
        cats = self.label_encoder_category.inverse_transform(top_indices_cat.cpu().numpy())
        confs_cat = top_probs_cat.cpu().numpy()
        top_predictions_cat = {s: float(c) for s, c in zip(cats, confs_cat)}
        
        # Top Sub-Category Predictions
        top_probs_subcat, top_indices_subcat = torch.topk(probs_subcat[0], k=min(3, len(self.label_encoder_subcategory.classes_)))
        subcats = self.label_encoder_subcategory.inverse_transform(top_indices_subcat.cpu().numpy())
        confs_subcat = top_probs_subcat.cpu().numpy()
        top_predictions_subcat = {s: float(c) for s, c in zip(subcats, confs_subcat)}

        prediction_result = {
            'category': {
                'prediction': predicted_category,
                'confidence': confidence_cat,
                'top_predictions': top_predictions_cat
            },
            'sub_category': {
                'prediction': predicted_subcategory,
                'confidence': confidence_subcat,
                'top_predictions': top_predictions_subcat
            }
        }
        
        return prediction_result
