import tkinter as tk
from pdb import Restart
from tkinter import Label, Button
import cv2
from PIL import Image, ImageTk
import torch
import numpy as np
import mediapipe as mp
import os
from dotenv import load_dotenv
import glob

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
                         'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
        random.shuffle(self.charList)

        self.currentList = self.charList
        self.currentIndex = 0
        self.incorrect = []
        self.retry_mode = False # track if were retrying
        # dont need to store correct

    def GetCurrentChar(self):
        # done when empty
        if not self.currentList:
            return "Done"
        if self.currentIndex < len(self.currentList):
            return self.currentList[self.currentIndex]
        return "Done"

    def CheckPred(self, predChar):
        target = self.GetCurrentChar()
        if target == "Done":
            return False

        predCharFormatted = str(predChar).upper()
        matching = (predCharFormatted == target)

        if not matching:
            # add to incorrect list if not in retry mode, keep it if we are
            if target not in self.incorrect:
                self.incorrect.append(target)

        return matching

    def SetNextChar(self):
        self.currentIndex += 1
        if self.currentIndex >= len(self.currentList):
            self.StartRetry()

    def StartRetry(self):
        if self.incorrect:
            print(f"[INFO] Main list done. Retrying {len(self.incorrect)} mistakes...")
            self.currentList = self.incorrect
            self.incorrect = []
            self.currentIndex = 0
            self.retry_mode = True
            random.shuffle(self.currentList) # shuffle mistakes
        else:
            self.currentList = []
            self.currentIndex = 0

    def Restart(self):
        print("[INFO] Restarting entire session...")
        self.currentList = self.charList
        random.shuffle(self.currentList)
        self.currentIndex = 0
        self.incorrect = []
        self.retry_mode = False

    def get_progress(self):
        if not self.currentList:
            return "Complete!"
        mode = "Retry" if self.retry_mode else "Test"
        return f"{self.currentIndex + 1}/{len(self.currentList)} ({mode})"


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

# select camera
def select_camera():
    # scan ports (max 3 who has more than 3)
    available_ports = []
    for i in range(3):
        temp_cap = cv2.VideoCapture(i)
        if temp_cap.isOpened():
            available_ports.append(i)
            temp_cap.release()
    if not available_ports:
        return 0
    if len(available_ports) == 1:
        return available_ports[0]

    selected_var = tk.IntVar(value=available_ports[0])

    dialog = tk.Toplevel(window)
    dialog.title("Select Camera")
    dialog.geometry("300x200")
    dialog.grab_set()

    Label(dialog, text="Multiple cameras found.\nPlease select one:", font=("Arial", 12)).pack(pady=15)

    for port in available_ports:
        tk.Radiobutton(dialog, text=f"Camera {port}", variable=selected_var, value=port, font=("Arial", 11)).pack(
            anchor="w", padx=80)

    def confirm():
        dialog.destroy()

    Button(dialog, text="Launch App", command=confirm, bg="#5cb85c", fg="white").pack(pady=20)

    # wait until selected
    window.wait_window(dialog)
    return selected_var.get()


# set camera from index rather than 0
selected_index = select_camera()
cap = cv2.VideoCapture(selected_index)
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
    if session.GetCurrentChar() == "Done":
        session.Restart()
        target_label.config(text=f"Target: {session.GetCurrentChar()}")
        progress_label.config(text=f"Progress: {session.get_progress()}")
        button.config(text="Take Picture")
        window.config(bg="SystemButtonFace")
        return

    ret, frame = cap.read()
    if ret:
        os.makedirs("./images", exist_ok=True)
        image_path = "./images/captured_image.jpg"
        cv2.imwrite(image_path, frame)

        try:
            prediction = run_model(image_path)
            # print model prediction
            print(f"[DEBUG] Raw Prediction: {prediction}")
            result = session.CheckPred(prediction)
            flash_background(result)
            session.SetNextChar()

            next_char = session.GetCurrentChar()
            if next_char == "Done":
                target_label.config(text="Session Complete!")
                progress_label.config(text="Click to Restart")
                button.config(text="Restart")
            else:
                target_label.config(text=f"Target: {next_char}")
                progress_label.config(text=f"Progress: {session.get_progress()}")

        except Exception as e:
            print("Error:", e)


show_frame()
window.mainloop()

cap.release()
cv2.destroyAllWindows()
