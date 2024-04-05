import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
import pickle
from sklearn.metrics import classification_report


# Define custom dataset class
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
# Define neural network model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    with open('X_train.pickle', 'rb') as file:
        X_train = pickle.load(file)
    with open('X_test.pickle', 'rb') as file:
        X_test = pickle.load(file)
    with open('X_val.pickle', 'rb') as file:
        X_val = pickle.load(file)
    with open('y_train.pickle', 'rb') as file:
        y_train = pickle.load(file)
    with open('y_test.pickle', 'rb') as file:
        y_test = pickle.load(file)
    with open('y_val.pickle', 'rb') as file:
        y_val = pickle.load(file)
    
    # Instantiate model, loss, and optimizer
    model = Model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Define DataLoader
    train_dataset = CustomDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_val)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_val).sum().item() / len(y_val)
        print(f'Accuracy on Val data: {accuracy}')
    
    print("Classification Report for DNN :\n", classification_report(y_val, predicted))
    
    torch.save(model, 'DNN.pt')
    print("DNN model saved!")
