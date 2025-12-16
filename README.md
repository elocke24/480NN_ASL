# ASL Language Learning App

Datasets in rotation:
- [Akash](https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data)
- [Debashish Sau](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset/data)

ASL Character Recognition Using MLP

Project Overview

This project implements a Multilayer Perceptron (MLP) to recognize and teach American Sign Language (ASL) alphabet characters using a camera-based application. The system captures images of hand signs, extracts hand landmark features using MediaPipe, and feeds those features into a trained MLP model to predict the corresponding ASL letter. Letters J and Z are excluded, as they require motion-based gestures. The application is intended as an educational tool to assist with learning and translating ASL characters in real time.


Technologies Used

Python
OpenCV (image capture and processing)
MediaPipe (hand landmark detection)
NumPy
Pillow (GUI image handling)
PyTorch (MLP model)
python-dotenv (.env configuration)
Tkinter (GUI)


Setup Instructions

After cloning the repository, install dependencies using pip install -r requirements.txt and set up the provided .env file which specifies the paths to the training and test data files, for example:
TRAINING_PATH=/path/to/training_data.csv
TEST_PATH=/path/to/test_data.csv


Running the Application

Run python main.py and ensure a working camera is connected before running the program.


Model Training (Optional)

If retraining the model:
1. Run the dataset preprocessing script.
2. Train the MLP using the provided training script.
3. Save the trained model to be loaded by the application.


Limitations

Does not support dynamic ASL letters J and Z
Performance depends on lighting and camera quality
Only supports one hand at a time