import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import classification_report


# Define the LSTM model
class LSTMNetwork(nn.Module):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __init__(self, input_size=106400, hidden_size=128, num_layers=2, output_size=2):
        super(LSTMNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, matrix, labels):
        self.x = torch.tensor(matrix.values, dtype=torch.float32).reshape(-1, 1, 106400)  # Assuming FastText embedding dimension is 300
        self.y = torch.tensor(labels.values, dtype=torch.long).reshape(-1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    


if __name__ == '__main__':
    with open('train_matrix.pickle', 'rb') as file:
        train_matrix = pickle.load(file)
    with open('test_matrix.pickle', 'rb') as file:
        test_matrix = pickle.load(file)
    with open('val_matrix.pickle', 'rb') as file:
        val_matrix = pickle.load(file)
    with open('train_labels.pickle', 'rb') as file:
        train_labels = pickle.load(file)
    with open('test_labels.pickle', 'rb') as file:
        test_labels = pickle.load(file)
    with open('val_labels.pickle', 'rb') as file:
        val_labels = pickle.load(file)
    
    train_dataset = CustomDataset(train_matrix, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = CustomDataset(val_matrix, val_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_dataset = CustomDataset(test_matrix, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    with open('test_dataloader.pickle', 'wb') as file:
        pickle.dump(test_dataloader, file)
    
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate the LSTM model
    lstm_network = LSTMNetwork().to(device)
    
    # Define criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lstm_network.parameters(), lr=0.001)
    
    # Training loop
    epochs = 10
    train_epoch_loss = []
    eval_epoch_loss = []
    
    for epoch in tqdm(range(epochs)):
        lstm_network.train()
        total_train_loss = 0.0
        total_train_samples = 0
        for train_x, train_y in train_dataloader:
            train_x = train_x.to(device)
            train_y = train_y.to(device)
            
            optimizer.zero_grad()
            
            outputs = lstm_network(train_x)
            loss = criterion(outputs, train_y)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item() * train_x.size(0)
            total_train_samples += train_x.size(0)
        
        train_epoch_loss.append(total_train_loss / total_train_samples)
    
        lstm_network.eval()
        total_eval_loss = 0.0
        total_eval_samples = 0
        for eval_x, eval_y in val_dataloader:
            eval_x = eval_x.to(device)
            eval_y = eval_y.to(device)
            
            with torch.no_grad():
                outputs = lstm_network(eval_x)
                loss = criterion(outputs, eval_y)
            
            total_eval_loss += loss.item() * eval_x.size(0)
            total_eval_samples += eval_x.size(0)
        
        eval_epoch_loss.append(total_eval_loss / total_eval_samples)
    
    plt.plot(range(epochs), train_epoch_loss, label='train')
    plt.plot(range(epochs), eval_epoch_loss, label='eval')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    correct = 0
    total = 0
    for x, y in val_dataloader:
        x = x.to(device)
        with torch.no_grad():
            yp = lstm_network(x)
        yp = torch.argmax(yp.cpu(), dim = 1)
        correct += (yp == y).sum()
        total += len(y)
    print(f"Accuracy on Val Data {(correct * 100 / total):.2f}")
    
    print("Classification Report for LSTM :\n", classification_report(y, yp))
    
    torch.save(lstm_network, 'LSTM.pt')
    print("LSTM model saved!")