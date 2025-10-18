import tkinter as tk
from tkinter import Label, Button
import cv2
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import transforms
import mediapipe as mp
import numpy as np
import os

#------------------------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)   	
        self.relu  = nn.ReLU()
        self.pool  = nn.MaxPool2d(2, 2)                           		
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  	
        self.fc1   = nn.Linear(32 * 32 * 32, 26)                    		

    def forward(self, x):
        x = self.relu(self.conv1(x))   
        x = self.pool(x)               	
        x = self.relu(self.conv2(x))   	
        x = self.pool(x)               	
        x = x.view(x.size(0), -1)      	
        x = self.fc1(x)                		
        return x
# model
def run_model(img):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)

    # Load model safely for CPU or GPU
    model.load_state_dict(torch.load("./asl_model.pth", map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])

    img = Image.open(img).convert("RGB")
    img = transform(img).unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

    letters = [chr(i + 65) for i in range(26)]  # A-Z
    prediction = letters[predicted.item()]

    print("Prediction:", prediction)
    print("Confidence:", probs[0][predicted.item()].item())
    print("Softmax probabilities:", probs[0].cpu().numpy())
#-----------------------------------------

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

def crop_hand(image):
    # Detects a hand and returns a cropped image around it.
    h, w, _ = image.shape
    results = hands_detector.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    if results.multi_hand_landmarks:
        # Take first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]

        # Convert normalized coords to pixel space
        x_min = int(min(x_coords) * w)
        x_max = int(max(x_coords) * w)
        y_min = int(min(y_coords) * h)
        y_max = int(max(y_coords) * h)

        # Add a little padding
        pad = int(0.3 * max(x_max - x_min, y_max - y_min))
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(w, x_max + pad)
        y_max = min(h, y_max + pad)

        cropped = image[y_min:y_max, x_min:x_max]
        return cropped
    else:
        print("No hand detected, using full frame.")
        return image

# Create window
window = tk.Tk()
window.title("Camera App")
window.geometry("800x600")  # starting size

# Open the default camera
cap = cv2.VideoCapture(0)

# Label for video feed
video_label = Label(window)
video_label.pack(fill="both", expand=True)  # allow it to resize with the window

def show_frame():
    ret, frame = cap.read()
    if ret:
        # Convert BGR (OpenCV) to RGB (Pillow)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize dynamically to current label size
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

        # Convert to ImageTk
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    video_label.after(10, show_frame)

def take_picture():
    ret, frame = cap.read()
    if ret:
        os.makedirs("./images", exist_ok=True)  # ensures folder exists

        # Detect and crop hand using MediaPipe
        cropped_hand = crop_hand(frame)

        image_path = "./images/captured_image.jpg"
        cv2.imwrite(image_path, cropped_hand)

        print("Image saved as captured_image.jpg")

        # Run the model on the cropped image
        run_model("./images/captured_image.jpg")

# Button on the video
button = Button(window, text="Take Picture", height=2, width=15, command=take_picture)
button.place(relx=0.5, rely=0.9, anchor='center')

# Start video loop
show_frame()

# Run the GUI loop
window.mainloop()

# Release the camera when done
cap.release()
cv2.destroyAllWindows()
