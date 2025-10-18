import tkinter as tk
from tkinter import Label, Button
import cv2
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import transforms
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
    # loading the model
    model = CNN()
    model.load_state_dict(torch.load("./asl_model.pth"))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])

    img = Image.open(img).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

    letters = [chr(i + 65) for i in range(26)]  # A-Z
    prediction = letters[predicted.item()]

    print("Prediction:", prediction)
    print("Confidence:", probs[0][predicted.item()].item())
    print("Softmax probabilities:", probs[0].numpy())
#-----------------------------------------

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
            frame = cv2.resize(frame, (label_width, label_height))

        # Convert to ImageTk
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    video_label.after(10, show_frame)

def take_picture():
    ret, frame = cap.read()
    if ret:
        os.makedirs("./images", exist_ok=True)  # <-- ensures folder exists
        image_path = "./images/captured_image.jpg"
        cv2.imwrite(image_path, frame)
        run_model("./images/captured_image.jpg")
        # print("Image saved as captured_image.jpg")

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
