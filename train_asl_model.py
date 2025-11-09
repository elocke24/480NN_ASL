"""
Train a simple MLP to classify ASL letters from hand landmarks.

Goal:
    Have a model that can accurately depict what ASL letter is being shown.

What the file does:
    - Reads training_path and test_path from a .env file.
    - Converts images to hand landmark points using MediaPipe.
    - Builds a dataset of landmark vectors and labels.
    - Trains a small MLP on the landmark data.
    - Tests the model on a set of training data and outputs accuracy.
    - Saves the trained model to disk.
"""

import sys
import os
import glob
import cv2
from dotenv import load_dotenv

import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Environment variables
load_dotenv() # reads .env file in current directory
TRAINING_PATH = os.getenv("training_path")
TEST_PATH = os.getenv("test_path")
MODEL_SAVE_PATH = os.getenv("model_path")

# Model hyperparameters
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 40
RANDOM_SEED = 42

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Landmark constants
NUM_LANDMARKS = 21  # MediaPipe hands return 21 landmarks
FEATURE_SIZE = NUM_LANDMARKS * 3  # (x, y, z) per landmark

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Dataset definition
class ASLDataset(Dataset):
    def __init__(self, root_dir, label_map):
        self.samples = []
        self.labels = []
        self.label_map = label_map
        self.hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

        # Gather all image paths first so we know total count
        # Limitation for this is that currently the images MUST be .jpg files!
        # Just keep this in mind if the script isn't working for you
        self.image_paths = []
        for label_name in label_map.keys():
            self.image_paths.extend(glob.glob(os.path.join(root_dir, label_name, "*.jpg")))

        total_images = len(self.image_paths)
        print(f"[INFO] Found {total_images} images to process.\n")
        
        # Process images with progress bar
        for idx, img_path in enumerate(self.image_paths, start=1):
            landmarks = self._extract_landmarks(img_path)

            if landmarks is not None:
                label_name = os.path.basename(os.path.dirname(img_path))
                self.samples.append(landmarks)
                self.labels.append(label_map[label_name])

            # Makeshift loading bar (updates in place)
            bar_length = 30  # length of the visual bar
            progress = idx / total_images
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            sys.stdout.write(f"\r[PROCESSING] |{bar}| {idx}/{total_images} images")
            sys.stdout.flush()

        print("\n[INFO] Landmark extraction complete.\n")
    
    def _extract_landmarks(self, img_path):
        # Read the image
        image = cv2.imread(img_path)
        if image is None:
            return None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        return np.array(landmarks, dtype=np.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
    
# Model defintion
class MLPClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Training and Evaluation
def train(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

# Main function breakdown
def main():
    if TRAINING_PATH is None or TEST_PATH is None:
        raise RuntimeError("Please set training_path and test_path in your .env file")
    
    print(f"Device: {DEVICE}")
    print(f"Loading training data from: {TRAINING_PATH}")
    print(f"Loading test data from: {TEST_PATH}")

    # Build the label map from training directory
    # This allows for more or less characters to added dynamically
    label_names = sorted(os.listdir(TRAINING_PATH))
    label_map = {name: idx for idx, name in enumerate(label_names)}
    
    print(f"Detected {len(label_map)} labels: {list(label_map.keys())}")

    # Populate the datamap, computing each picture and set to a letter
    # Each label in the map should have the LANDMARK DATA in each of its
    # entires, NOT the image!
    train_dataset = ASLDataset(TRAINING_PATH, label_map)
    test_dataset = ASLDataset(TEST_PATH, label_map)

    # For Mac users: make sure you delete the .DS_Store files in your dataset
    # otherwise this will output one more directory than expected
    # One way to do this is to cd into the directory where the labels directories
    # are, and run the terminal command: "find . -name ".DS_Store" -type f -delete"
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Instantiate the model and load the data into it
    # First, load the data
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Setup the model
    model = MLPClassifier(FEATURE_SIZE, len(label_map)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Train the model (the loop)
    for epoch in range(EPOCHS):
        loss = train(model, train_loader, criterion, optimizer)
        acc = evaluate(model, test_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss:.4f}, Test Acc: {acc:.4f}")


    # Save the model
    torch.save({
        "model_state_dict": model.state_dict(),
        "label_map": label_map
    }, MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    print("Done.")




if __name__ == "__main__":
    main()