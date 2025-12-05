import tkinter as tk
from tkinter import Label, Button
import cv2
from PIL import Image, ImageTk
import torch
import numpy as np
import mediapipe as mp
import os
from dotenv import load_dotenv

# ---------------------------
# ASLLearning class
# ---------------------------
import random


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

class ASLLearning:
    def __init__(self):
        self.charList = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                         'O', 'P', 'Q', 'R', 'S', 'T', 'U','V', 'W', 'X', 'Y']
        self.currentList = self.charList
        self.currentIndex = 0
        self.correct = []
        self.incorrect = []

    def ShuffleList(self):
        random.shuffle(self.currentList)

    def ResetList(self):
        random.shuffle(self.currentList)
        self.currentIndex = 0

    def GetCurrentChar(self):
        if self.currentIndex < len(self.currentList):
            return self.currentList[self.currentIndex]
        return None

    def SetNextChar(self):
        self.currentIndex += 1

    def GetNextChar(self):
        if self.currentIndex + 1 < len(self.currentList):
            return self.currentList[self.currentIndex + 1]
        return None

    def CheckPred(self, predChar):
        target = self.GetCurrentChar()
        if target is None:
            return False

        predCharFormatted = str(predChar).upper()
        matching = (predCharFormatted == target)

        if matching:
            self.correct.append(target)
        else:
            self.incorrect.append(target)

        return matching

    def clear_incorrect_list(self):
        self.incorrect = []

    def clear_correct_list(self):
        self.correct = []

    def get_progress(self):
        return f"{self.currentIndex + 1}/{len(self.currentList)}"


# Create ASL session
session = ASLLearning()


# ---------------------------
# Load environment vars
# ---------------------------
load_dotenv()
MODEL_SAVE_PATH = os.getenv("model_path", "asl_mlp_model.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mp_hands = mp.solutions.hands

# ---------------------------
# MLP model
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
# Load model
# ---------------------------
def load_model():
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    label_map = checkpoint["label_map"]
    model = MLPClassifier(input_size=63, num_classes=len(label_map))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    inv_label_map = {v: k for k, v in label_map.items()}
    return model, inv_label_map

model, inv_label_map = load_model()

# ---------------------------
# Extract landmarks
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
# Predict letter
# ---------------------------
def run_model(img_path):
    landmarks = extract_landmarks(img_path)
    landmarks = normalize_landmarks(landmarks)
    x = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(x)
        _, pred = torch.max(outputs, 1)
        label = inv_label_map[pred.item()]

    print(f"[RESULT] Predicted ASL Letter: {label}")
    return label


# ---------------------------
# GUI
# ---------------------------
window = tk.Tk()
window.title("ASL Camera App")
window.geometry("800x600")

cap = cv2.VideoCapture(0)

# ---------------------------
# BOTTOM BAR (CENTERED)
# ---------------------------
bottom_frame = tk.Frame(window)
bottom_frame.pack(side="bottom", fill="x", pady=10)

inner = tk.Frame(bottom_frame)
inner.pack(expand=True)   # <--- THIS centers the whole bar

target_label = Label(inner, text=f"Target: {session.GetCurrentChar()}", font=("Arial", 16))
target_label.pack(side="left", padx=40)

button = Button(inner, text="Take Picture", height=2, width=15, command=lambda: take_picture())
button.pack(side="left", padx=40)

progress_label = Label(inner, text=f"Progress: {session.get_progress()}", font=("Arial", 16))
progress_label.pack(side="left", padx=40)

# define after so its not drawn over
video_label = Label(window)
video_label.pack(fill="both", expand=True)

# ---------------------------
# Video Loop
# ---------------------------
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


# ---------------------------
# Capture + Predict
# ---------------------------
def flash_background(is_correct):
    color = "green" if is_correct else "red"
    window.config(bg=color)

    # Restore default after 2 seconds
    window.after(2000, lambda: window.config(bg="SystemButtonFace"))

def take_picture():
    ret, frame = cap.read()
    if ret:
        os.makedirs("./images", exist_ok=True)
        image_path = "./images/captured_image.jpg"
        cv2.imwrite(image_path, frame)

        try:
            prediction = run_model(image_path)
            result = session.CheckPred(prediction)
            flash_background(result)
            session.SetNextChar()

            target_label.config(text=f"Target: {session.GetCurrentChar()}")
            progress_label.config(text=f"Progress: {session.get_progress()}")

        except Exception as e:
            print("Error:", e)


show_frame()
window.mainloop()

cap.release()
cv2.destroyAllWindows()
