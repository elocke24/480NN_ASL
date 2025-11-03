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

# Main function breakdown
def main():
    if TRAINING_PATH is None or TEST_PATH is None:
        raise RuntimeError("Please set training_path and test_path in your .env file")
    
    print(f"Device: {DEVICE}")
    print(f"Loading training data from: {TRAINING_PATH}")
    print(f"Loading test data from: {TEST_PATH}")

    # Build the label map from training directory
    # This allows for more or less characters to added dynamically
    print("Dectected XX labels: LIST_OF_LABELS_HERE")

    # Populate the datamap, computing each picture and set to a letter
    # Each label in the map should have the LANDMARK DATA in each of its
    # entires, NOT the image!
    print("Training samples: LENGTH")
    print("Test samples: LENGTH")

    # Instantiate the model and load the data into it

    # Train the model

    # Save the model

    print("Done.")




if __name__ == "__main__":
    main()