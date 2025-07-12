import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os

DATASET_PATH = 'dataset/train'
if os.path.isdir(DATASET_PATH):
    class_names = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
else:
    class_names = ['eating', 'idle', 'playing', 'sleeping']


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Classifier using device: {device}")

try:
    model = torch.load("cat_classifier.pth", weights_only=False)
    model = model.to(device)
    model.eval()
    print("Classifier model loaded successfully.")
except FileNotFoundError:
    print("Error: 'cat_classifier.pth' not found. Please run train.py first.")
    model = None


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def classify(cat_roi):
    if model is None:
        return "Model not loaded "
    
    image = cv2.cvtColor(cat_roi, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    input_tensor = transform(image).unsqueeze(0)

    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred_index = output.argmax(dim=1).item()

    return class_names[pred_index]
