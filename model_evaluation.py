import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config

def evaluate_model():
    # Define transform
    transform = transforms.Compose([
        transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])

    # Load test set
    try:
        test_dataset = datasets.ImageFolder(root=config.TEST_PATH, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)
    except Exception as e:
        print(f"Error loading test dataset: {str(e)}")
        return

    # Load model
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(config.DROPOUT_RATE),
        nn.Linear(num_ftrs, config.NUM_CLASSES)
    )
    try:
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE, weights_only=True))
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    model = model.to(config.DEVICE)
    model.eval()

    # Get predictions
    all_preds = []
    all_labels = []
    with torch.no_grad():
        try:
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            return

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)
    accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_names = list(config.CLASS_NAMES.values())

    # Print results
    print("\nTest Set Evaluation:")
    print(f"Accuracy: {accuracy:.2f}%")
    print("Per-class Metrics:")
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        print(f"Class {i} ({class_names[i]}): Precision={p:.2f}, Recall={r:.2f}, F1={f:.2f}")
    print("Confusion Matrix (Numerical):")
    print(conf_matrix)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    try:
        plt.savefig(os.path.join(config.BASE_PATH, "confusion_matrix.png"))
        plt.close()
        print("Confusion matrix saved as 'confusion_matrix.png'")
    except Exception as e:
        print(f"Error saving confusion matrix: {str(e)}")

if __name__ == "__main__":
    evaluate_model()