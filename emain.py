
import tkinter as tk
from tkinter import Label, Button
import cv2
from PIL import Image, ImageTk
import torch
import numpy as np
import mediapipe as mp
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MODEL_SAVE_PATH = os.getenv("model_path", "asl_mlp_model.pt")

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands

# ---------------------------
# MLP model definition (must match training)
# ---------------------------
class MLPClassifier(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------
# Load trained model
# ---------------------------
def load_model():
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    label_map = checkpoint["label_map"]
    model = MLPClassifier(input_size=63, num_classes=len(label_map))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    inv_label_map = {v: k for k, v in label_map.items()}  # reverse lookup
    return model, inv_label_map

model, inv_label_map = load_model()

# ---------------------------
# Extract landmarks from an image
# ---------------------------
def extract_landmarks(image_path):
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if not results.multi_hand_landmarks:
        raise ValueError("No hand detected in the image.")

    hand_landmarks = results.multi_hand_landmarks[0]
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return np.array(landmarks, dtype=np.float32)

# ---------------------------
# Predict ASL letter
# ---------------------------
def run_model(img_path):
    landmarks = extract_landmarks(img_path)
    x = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        _, pred = torch.max(outputs, 1)
        label = inv_label_map[pred.item()]

    letters = [chr(i + 65) for i in range(len(inv_label_map))]

    # Print all softmax probabilities nicely
    print("[Softmax Probabilities]")
    for ltr, p in zip(letters, probs[0].cpu().numpy()):
        print(f"{ltr}: {p:.4f}")

    print(f"[RESULT] Predicted ASL Letter: {label}")
    return label

# ---------------------------
# GUI setup
# ---------------------------
window = tk.Tk()
window.title("ASL Camera App")
window.geometry("800x600")

cap = cv2.VideoCapture(0)

video_label = Label(window)
video_label.pack(fill="both", expand=True)

def show_frame():
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        label_width = video_label.winfo_width()
        label_height = video_label.winfo_height()
        if label_width > 0 and label_height > 0:
            frame_height, frame_width = frame.shape[:2]
            frame_aspect = frame_width / frame_height
            label_aspect = label_width / label_height
            if label_aspect > frame_aspect:
                new_height = label_height
                new_width = int(new_height * frame_aspect)
            else:
                new_width = label_width
                new_height = int(new_width / frame_aspect)
            if new_width > 0 and new_height > 0:
                frame = cv2.resize(frame, (new_width, new_height))
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
    video_label.after(10, show_frame)

def take_picture():
    ret, frame = cap.read()
    if ret:
        os.makedirs("./images", exist_ok=True)
        image_path = "./images/captured_image.jpg"
        cv2.imwrite(image_path, frame)
        try:
            prediction = run_model(image_path)
            print(f"Predicted Letter: {prediction}")
        except Exception as e:
            print("Error:", e)

button = Button(window, text="Take Picture", height=2, width=15, command=take_picture)
button.place(relx=0.5, rely=0.9, anchor='center')

show_frame()
window.mainloop()

cap.release()
cv2.destroyAllWindows()
