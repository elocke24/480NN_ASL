# loading a pre-trained model off gpt - resnet18
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from PIL import Image
from dotenv import load_dotenv
import os

# -------------------------
# Config
# -------------------------
# enviroment variables
load_dotenv()
train_path = os.getenv("train_path")
test_path = os.getenv("test_path")
BATCH_SIZE = 32
EPOCHS = 5
LR = 0.001
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# -------------------------
# Transforms
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),          # ResNet input size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------
# Load Datasets
# -------------------------
train_data = datasets.ImageFolder(train_path, transform=transform)
test_data  = datasets.ImageFolder(test_path,  transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE)

print(f"Classes: {train_data.classes}")

# -------------------------
# Load Pretrained Model
# -------------------------
model = models.resnet18(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final fully connected layer
num_classes = len(train_data.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

# -------------------------
# Loss and Optimizer
# -------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)

# -------------------------
# Training Loop
# -------------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}")

# -------------------------
# Evaluate on Test Set
# -------------------------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# -------------------------
# Predict Single Image
# -------------------------
def predict_image(img_path):
    img = Image.open(img_path)
    img_t = transform(img).unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        output = model(img_t)
        _, pred = torch.max(output, 1)
    print(f"Predicted class for {img_path}: {train_data.classes[pred.item()]}")

# Example usage
# predict_image("data/test/A/sample1.jpg")
