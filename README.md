# ASL Language Learning App

Datasets in rotation:
- [Akash](https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data)
- [Debashish Sau](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset/data)

ASL Character Recognition Using MLP

Project Overview

This project implements a Multilayer Perceptron (MLP) to recognize and teach American Sign Language (ASL) alphabet characters using a camera-based application. The system captures images of hand signs, extracts hand landmark features using MediaPipe, and feeds those features into a trained MLP model to predict the corresponding ASL letter. Letters J and Z are excluded, as they require motion-based gestures. The application is intended as an educational tool to assist with learning and translating ASL characters in real time.

Technologies Used

Python 3.10
OpenCV (image capture and processing)
MediaPipe (hand landmark detection)
NumPy
Pillow (GUI image handling)
PyTorch (MLP model)
python-dotenv (.env configuration)

# Setup Instructions
Ensure Python 3.10 is installed (mediapipe will not work with a newer version)
cd into your desired folder to install the project, then run the following commands:
``git clone https://github.com/elocke24/480NN_ASL.git
cd 480NN_ASL``

after this, set up a virtual enviornment (depends on your opperating system)
Windows:
``py -3.10 -m venv venv
venv\Scripts\activate``
If re-running, ``venv\Scripts\activate.bat``

Mac/Linux
``python3.10 -m venv venv
source venv/bin/activate``

Ensure the virtual enviornment is activated and install dependancies
``pip install -r requirements.txt``


If you want to train / test the model (Optional)
Log into your emich account and download the .env file and data.zip folder from this link:
https://drive.google.com/drive/folders/1Up0EY59YKFChX36pa14S3Bq9iNLENNdc?usp=sharing

Unzip data.zip and place the unzipped data folder and .env file into the 480NN_ASL folder
**Ensure the downloaded env file is .env not env, it likes to be changed when downloaded**
Run ``train_asl_model.py`` (this will take a long time)
This will create / override ``asl_mlp_model.pt``

# Running the Application
ensure a working camera is connected before running the program
Run ``python emain.py``


Limitations

Does not support dynamic ASL letters J and Z
Performance depends on lighting and camera quality
Only supports one hand at a time
