import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import classification_report

# If you're using PyTorch, you can create a custom dataset to handle this
class CustomDataset(Dataset):
    def __init__(self, matrix, labels):
        self.x = torch.tensor(matrix.values, dtype=torch.float32).reshape(-1,1,106400)
        self.y = torch.tensor(labels.values, dtype=torch.long).reshape(-1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class Network(nn.Module):
  def __init__(self, input_channel = 1, output_dim = 2):
    super().__init__()
    self.conv_1 = nn.Conv1d(1, 128, 28, stride = 28)
    self.activation_1 = nn.ReLU()

    self.conv_2 = nn.Conv1d(128, 256, 28, stride = 28)
    self.activation_2 = nn.ReLU()

    self.flatten = nn.Flatten()

    self.linear_3 = nn.Linear(34560, output_dim)

  def forward(self, x):
    out = self.conv_1(x)
    out = self.activation_1(out)

    out = self.conv_2(out)
    out = self.activation_2(out)

    out = self.flatten(out)

    out = self.linear_3(out)

    return out
  

  


if __name__ == "__main__":
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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    network = Network().to(device)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(network.parameters(), lr = 0.001)
    epochs = 10

    train_epoch_loss = []
    eval_epoch_loss = []

    for epoch in tqdm(range(epochs)):
        curr_loss = 0
        total = 0
        for train_x, train_y in train_dataloader:
            train_x = train_x.to(device)
            train_y = train_y.to(device)
            optim.zero_grad()

            y_pred = network(train_x)
            loss = criterion(y_pred, train_y)

            loss.backward()
            optim.step()

            curr_loss += loss.item()
            total += len(train_y)
        train_epoch_loss.append(curr_loss / total)

        curr_loss = 0
        total = 0
        for eval_x, eval_y in val_dataloader:
            eval_x = eval_x.to(device)
            eval_y = eval_y.to(device)
            optim.zero_grad()

            with torch.no_grad():
                y_pred = network(eval_x)

            loss = criterion(y_pred, eval_y)

            curr_loss += loss.item()
            total += len(train_y)
        eval_epoch_loss.append(curr_loss / total)



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
            yp = network(x)
        yp = torch.argmax(yp.cpu(), dim = 1)
        correct += (yp == y).sum()
        total += len(y)
    print(f"Accuracy on Val Data {(correct * 100 / total):.2f}")

    print("Classification Report for CNN :\n", classification_report(y, yp))

    torch.save(network, 'CNN.pt')
    print("CNN model saved!")