from train_asl_model import MLPClassifier
import torch

def main():
    # Basic structure to just test the model below

    # Load model in from the directory in the .env file
    model = torch.load()

    # Load up UI:
    #   - Camera
    #   - Button to take a picture

    # Button logic:
    #   - Save specific frame to memory 
    #   - Also save it directly to a file in order to bug test
    #   - Run the image through MediaPipe to get the landmark points
    #   - Run the landmark points through the model
    #   - Output prediction to console

    return 0

if __name__ == "__main__":
    main()