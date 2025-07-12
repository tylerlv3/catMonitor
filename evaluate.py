import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = "cat_classifier.pth"
DATASET_PATH = "dataset"
BATCH_SIZE = 32

def evaluate_model():
    """
    Loads the trained model and evaluates it on the validation set,
    generating and saving a confusion matrix and classification report.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    try:
        model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.eval()
        print(f"Model loaded successfully from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Model file not found at '{MODEL_PATH}'.")
        return
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    val_dir = os.path.join(DATASET_PATH, "val")
    if not os.path.isdir(val_dir):
        print(f"Error: Validation directory not found at '{val_dir}'")
        return

    print("Loading validation dataset...")
    image_dataset = datasets.ImageFolder(val_dir, data_transform)
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    class_names = image_dataset.classes
    print(f"Found class names: {class_names}")

    all_preds = []
    all_labels = []

    print("Generating predictions on validation set...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n--- Evaluation Results ---")

    print("Calculating and saving confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("Saved confusion_matrix.png")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

if __name__ == "__main__":
    evaluate_model()