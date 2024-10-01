import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

class Simple_Neural_Network(nn.Module):
    def __init__(self):
        super(Simple_Neural_Network, self).__init__()
        self.fully_connected1 = nn.Linear(30, 16)  
        self.fully_connected2 = nn.Linear(16, 2)  

    def forward(self, x):
        x = torch.relu(self.fully_connected1(x))
        x = torch.log_softmax(self.fully_connected2(x), dim=1)
        return x

class Complex_Neural_Network(nn.Module):
    def __init__(self):
        super(Complex_Neural_Network, self).__init__()
        self.fully_connected1 = nn.Linear(30, 64)
        self.fully_connected2 = nn.Linear(64, 32)
        self.fully_connected3 = nn.Linear(32, 2)

    def forward(self, x):
        x = torch.relu(self.fully_connected1(x))
        x = torch.relu(self.fully_connected2(x))
        x = torch.log_softmax(self.fully_connected3(x), dim=1)
        return x

def train_and_evaluate(model, train_loader, test_loader, epochs=15):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{epochs} - Loss: {loss.item()}')

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy}%')

Simple_Neural_Network_model = Simple_Neural_Network()
Complex_Neural_Network_model = Complex_Neural_Network()

print("Training Simple Neural Network:")
train_and_evaluate(Simple_Neural_Network_model, train_loader, test_loader)

print("\nTraining Complex Neural Network:")
train_and_evaluate(Complex_Neural_Network_model, train_loader, test_loader)
