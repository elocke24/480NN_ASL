"""
This file aims to check the realtime accuracy of whatever model is
generated from train_asl_model.py.

This file requires that there exists a file in the working directory
labeled, "asl_mlp_model.pt", where the model data can be loaded from.

What the file does:
    - Load in the model
    - Spin up a webcam
    - Detects a hand, overlays it with the landmarks, and outputs a label on screen
      in realtime

~Pressing 'q' on the keyboard quits the script
"""

import os
import numpy as np
import cv2
import torch
from dotenv import load_dotenv

from train_asl_model import (
    MLPClassifier,
    MODEL_SAVE_PATH,
    DEVICE,
    FEATURE_SIZE,
    mp_hands,
    mp_drawing,
)

def extract_landmarks_from_frame(image, hands_processor):
    """Extract MediaPipe hand landmarks from a frame and return results."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_processor.process(image_rgb)
    if not results.multi_hand_landmarks:
        return None, results  # Return both for drawing logic below

    hand_landmarks = results.multi_hand_landmarks[0]
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return np.array(landmarks, dtype=np.float32), results


def main():

    # Debugging main function...

    # 1. Load in the model
    # 2. Spin up the camera
    #   - Live video feed
    #   - Label in the upper-left corner with the prediction
    #   - Landmark points returned by MediaPipe drawn on the hand
    # 3. Contine the video loop until 'q' is pressed


    load_dotenv()
    model_path = MODEL_SAVE_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}")

    checkpoint = torch.load(model_path, map_location=DEVICE)
    label_map = checkpoint["label_map"]
    model = MLPClassifier(FEATURE_SIZE, len(label_map)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    inv_label_map = {v: k for k, v in label_map.items()}

    print(f"[INFO] Loaded model from {model_path}")
    print(f"[INFO] Labels: {list(label_map.keys())}")
    print("[INFO] Starting live ASL classification...")
    print("[INFO] Press 'q' to quit.\n")

    # Initialize camera and MediaPipe
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
    predicted_label = "Waiting..."
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame not captured.")
            break

        frame_count += 1

        # Run MediaPipe every few frames for performance
        if frame_count % 3 == 0:
            landmarks, results = extract_landmarks_from_frame(frame, hands)
            if landmarks is not None:
                with torch.no_grad():
                    x = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    outputs = model(x)
                    _, pred = torch.max(outputs, 1)
                    predicted_label = inv_label_map[pred.item()]
            else:
                predicted_label = "No hand detected"
        else:
            # Get last results to keep displaying landmarks even between frames
            _, results = extract_landmarks_from_frame(frame, hands)

        # Draw landmarks if detected
        if results and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2),
                )

        # Overlay prediction text
        cv2.putText(
            frame,
            f"Prediction: {predicted_label}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("ASL Live Classifier", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
