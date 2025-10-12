import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score


transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_ds = datasets.ImageFolder(root="./data/asl_alphabet_train/asl_alphabet_train", transform=transform)
test_ds = datasets.ImageFolder(root="./data/asl_alphabet_test/asl_alphabet_test", transform=transform)

batch_size = 64
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)   	
        self.relu  = nn.ReLU()
        self.pool  = nn.MaxPool2d(2, 2)                           		
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  	
        self.fc1   = nn.Linear(32 * 8 * 8, 26)                    		

    def forward(self, x):
        x = self.relu(self.conv1(x))   
        x = self.pool(x)               	
        x = self.relu(self.conv2(x))   	
        x = self.pool(x)               	
        x = x.view(x.size(0), -1)      	
        x = self.fc1(x)                		
        return x
    
def train_model(model, train_loader, test_loader, num_epochs=5, lr=1e-3):
    print("Starting training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

         # Within the same def train_model function

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            all_preds.extend(predicted.tolist())
            all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {acc * 100:.2f}%")

cnn_model = CNN()
train_model(cnn_model, train_loader, test_loader)

torch.save(cnn_model.state_dict(), 'asl_model.pth') # Save model
