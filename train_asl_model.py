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

import os
from dotenv import load_dotenv

import mediapipe as mp
import torch

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