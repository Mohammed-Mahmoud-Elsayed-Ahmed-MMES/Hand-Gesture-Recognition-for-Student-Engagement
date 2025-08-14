import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import os
import config

def train_model():
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])

    # Load datasets
    try:
        train_dataset = datasets.ImageFolder(root=config.TRAIN_PATH, transform=transform)
        val_dataset = datasets.ImageFolder(root=config.VAL_PATH, transform=transform)
    except Exception as e:
        print(f"Error loading datasets: {str(e)}")
        return None, None

    # Class weights for imbalance
    class_weights = [1.0 / count for count in config.CLASS_COUNTS]
    weights = [class_weights[label] for _, label in train_dataset.samples]
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(train_dataset), replacement=True)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)

    # Model setup
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(config.DROPOUT_RATE),
        nn.Linear(num_ftrs, config.NUM_CLASSES)
    )
    model = model.to(config.DEVICE)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    # Early stopping
    best_val_acc = 0.0
    patience = config.PATIENCE_EARLY_STOPPING
    trigger_times = 0

    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        try:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
        except Exception as e:
            print(f"Error during training epoch {epoch+1}: {str(e)}")
            continue

        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            try:
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            except Exception as e:
                print(f"Error during validation epoch {epoch+1}: {str(e)}")
                continue

        val_acc = 100 * correct / total
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Validation Accuracy: {val_acc:.2f}%, Val Loss: {avg_val_loss:.4f}')

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Early stopping and model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            trigger_times = 0
            try:
                os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
                torch.save(model.state_dict(), config.MODEL_PATH)
                print(f"Saved best model with accuracy: {best_val_acc:.2f}%")
            except Exception as e:
                print(f"Error saving model: {str(e)}")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping triggered after epoch {epoch+1}")
                break

    return model, config.MODEL_PATH

if __name__ == "__main__":
    model, best_model_path = train_model()
    if model is not None:
        print(f"Training completed. Best model saved at {best_model_path}")