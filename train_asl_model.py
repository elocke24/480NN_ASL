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
from torch.utils.data import Dataset, DataLoader, random_split
import random

# Environment variables
load_dotenv() # reads .env file in current directory
TRAINING_PATH = os.getenv("training_path")
TEST_PATH = os.getenv("test_path")
MODEL_SAVE_PATH = os.getenv("model_path", "asl_mlp_model.pt")

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


def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 3)

    # center based on wrist
    wrist = landmarks[0]
    landmarks -= wrist

    # scale normalization
    max_value = np.max(np.abs(landmarks))
    if max_value > 0:
        landmarks /= max_value

    return landmarks.flatten()


def augment_landmarks(landmarks):
    data = landmarks.reshape(-1, 3)

    # random rotation
    theta = np.radians(np.random.uniform(-10, 10))
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
    data = np.dot(data, rotation_matrix)

    # small random noise
    noise = np.random.normal(0, 0.02, data.shape)
    data += noise

    return data.flatten().astype(np.float32)

# Dataset definition
class ASLDataset(Dataset):
    # accepts a list of file_paths instead of finding them itself
    def __init__(self, image_paths, label_map, augment=False):
        self.samples = []
        self.labels = []
        self.label_map = label_map
        self.augment = augment
        self.hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

        total_images = len(image_paths)
        print(f"[INFO] Processing {total_images} images (Augment={augment})...")

        for idx, img_path in enumerate(image_paths, start=1):
            landmarks = self._extract_landmarks(img_path)
            if landmarks is not None:
                # get label from folder name (folder name is letter name if this changes we're cooked)
                label_name = os.path.basename(os.path.dirname(img_path))
                if label_name in label_map:
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
        print(f"\n[INFO] Complete. Valid samples: {len(self.samples)}")
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

        return normalize_landmarks(landmarks)

    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        # grab the clean normalized landmarks
        landmarks = self.samples[idx]
        if self.augment:
            landmarks = augment_landmarks(landmarks)

        x = torch.tensor(landmarks, dtype=torch.float32)
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
    if TRAINING_PATH is None:
        raise RuntimeError("Please set training_path and test_path in your .env file")
    label_names = sorted(os.listdir(TRAINING_PATH))
    label_map = {name: idx for idx, name in enumerate(label_names)}
    print(f"Labels: {list(label_map.keys())}")

    # For Mac users: make sure you delete the .DS_Store files in your dataset
    # otherwise this will output one more directory than expected
    # One way to do this is to cd into the directory where the labels directories
    # are, and run the terminal command: "find . -name ".DS_Store" -type f -delete"

    # gather all the files
    train_paths = []
    test_paths = []

    print("Stratifying dataset...")

    for label in label_names:
        folder_path = os.path.join(TRAINING_PATH, label)
        # get all jpgs
        files = glob.glob(os.path.join(folder_path, "*.jpg"))
        random.shuffle(files)

        # split
        split_idx = int(0.8 * len(files))
        train_paths.extend(files[:split_idx])
        test_paths.extend(files[split_idx:])

    # snuffle
    random.shuffle(train_paths)
    random.shuffle(test_paths)

    # make datasets
    print(f"Total Training: {len(train_paths)}")
    print(f"Total Testing: {len(test_paths)}")

    # Populate the datamap, computing each picture and set to a letter
    # Each label in the map should have the LANDMARK DATA in each of its
    # entries, NOT the image!
    print("--- Loading Training Data ---")
    train_dataset = ASLDataset(train_paths, label_map, augment=True)

    print("--- Loading Test Data ---")
    test_dataset = ASLDataset(test_paths, label_map, augment=False)

    # 5. Loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

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
    torch.save({"model_state_dict": model.state_dict(), "label_map": label_map}, MODEL_SAVE_PATH)
    print("Done.")




if __name__ == "__main__":
    main()