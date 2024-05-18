import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize((0.5,), (0.5,))  
])

train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

class fully_connected(nn.Module):
    def __init__(self):
        super(fully_connected, self).__init__()
        self.fully_connected1 = nn.Linear(784, 256)  
        self.fully_connected2 = nn.Linear(256, 128)  
        self.fully_connected3 = nn.Linear(128, 10)  

    def forward(self, x):
        x = x.view(-1, 28 * 28)  
        x = torch.relu(self.fully_connected1(x))
        x = torch.relu(self.fully_connected2(x))
        return torch.log_softmax(self.fully_connected3(x), dim=1)

class convolutional(nn.Module):
    def __init__(self):
        super(convolutional, self).__init__()
        self.convolutional1 = nn.Conv2d(1, 32, 3, padding=1)  
        self.convolutional2 = nn.Conv2d(32, 64, 3, padding=1) 
        self.pool = nn.MaxPool2d(2, 2)
        self.fully_connected1 = nn.Linear(64 * 7 * 7, 128)       
        self.fully_connected2 = nn.Linear(128, 10)            

    def forward(self, x):
        x = self.pool(torch.relu(self.convolutional1(x)))
        x = self.pool(torch.relu(self.convolutional2(x)))
        x = x.view(-1, 64 * 7 * 7) 
        x = torch.relu(self.fully_connected1(x))
        return torch.log_softmax(self.fully_connected2(x), dim=1)

def train(model, train_loader, criterion, optimizer, epochs=15):
    model.train()
    for epoch in range(epochs):
        running_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader)}')

def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy}%')
    return accuracy

fully_connected_model = fully_connected()
criterion = nn.NLLLoss()
optimizer_fully_connected = optim.Adam(fully_connected_model.parameters(), lr=0.003)
train(fully_connected_model, train_loader, criterion, optimizer_fully_connected)
accuracy_fully_connected = test(fully_connected_model, test_loader)

convolutional_model = convolutional()
optimizer_cnn = optim.Adam(convolutional_model.parameters(), lr=0.003)
train(convolutional_model, train_loader, criterion, optimizer_cnn)
accuracy_convolutional_model = test(convolutional_model, test_loader)
