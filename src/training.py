"""
Training and evaluation utilities
"""
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np


class Trainer:
    """Trainer class for model training and evaluation"""
    
    def __init__(self, model, device='cpu', learning_rate=1e-3, num_classes=10):
        """
        Args:
            model: PyTorch model
            device (str or torch.device): Device to train on
            learning_rate (float): Learning rate
            num_classes (int): Number of classes
        """
        self.model = model
        # Ensure device is a torch.device object
        if isinstance(device, str):
            if device == 'cuda' and not torch.cuda.is_available():
                device = 'cpu'
            self.device = torch.device(device)
        else:
            self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        self.num_classes = num_classes
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for images, labels in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({
                'Loss': running_loss / (total / 32),
                'Acc': 100 * correct / total
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def evaluate(self, val_loader):
        """Evaluate model on validation set"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Evaluating")
            
            for images, labels in progress_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, num_epochs=10, save_path='best_model.pth'):
        """Complete training loop"""
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 5
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.evaluate(val_loader)
            
            # Record history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"Model saved with val_loss: {val_loss:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print("Early stopping triggered")
                    break
        
        return history
    
    def predict(self, image_loader):
        """Get predictions for a batch of images"""
        self.model.eval()
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for images, _ in image_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(predictions), np.array(probabilities)
