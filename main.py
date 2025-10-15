import tkinter as tk
from tkinter import Label, Button
import cv2
from PIL import Image, ImageTk
import torch
from torchvision import transforms

#------------------------------
# model
def run_model(img):
    # loading the model
    model = torch.load("./asl_model.pth", weights_only=False)
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
        _, predicted = torch.max(outputs, 1)

    print("Prediction:", predicted.item())
#-----------------------------------------

# Create window
window = tk.Tk()
window.title("Camera App")

# Open the default camera
cap = cv2.VideoCapture(0)

# Label to display the video feed
video_label = Label(window)
video_label.pack()

def show_frame():
    ret, frame = cap.read()
    if ret:
        # Convert BGR (OpenCV) to RGB (Pillow)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
    video_label.after(10, show_frame)

def take_picture():
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("./images/captured_image.jpg", frame)
        run_model("./images/captured_image.jpg")
        # print("Image saved as captured_image.jpg")

# Button to take picture
Button(window, height=30, width=50, text="Take Picture", command=take_picture).pack()

# Start video loop
show_frame()

# Run the GUI loop
window.mainloop()

# Release the camera when done
cap.release()
cv2.destroyAllWindows()
