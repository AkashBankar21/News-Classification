import sys
import pickle
import torch
from LSTM import CustomDataset, LSTMNetwork
from CNN import Network
from DNN import Model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, f1_score, recall_score

# Check if arguments are provided
if len(sys.argv) < 3:
    print("Usage: python script.py <path> <model>")
    sys.exit(1)

# Access command-line arguments
path = str(sys.argv[1])
model1 = sys.argv[2]

with open('test_dataloader.pickle', 'rb') as file:
    test_dataloader = pickle.load(file)
with open('X_test.pickle', 'rb') as file:
    X_test = pickle.load(file)
with open('y_test.pickle', 'rb') as file:
    y_test = pickle.load(file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load(path)

def report(y_test, predicted):
    print("Classification Report :\n", classification_report(y_test, predicted))
    print("Precision : ", precision_score(y_test, predicted))
    print("Recall : ", recall_score(y_test, predicted))
    print("F1-Score : ", f1_score(y_test, predicted))
    print("Confusion Matrix : ", confusion_matrix(y_test, predicted))

if model1 == 'DNN':
    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).sum().item() / len(y_test)
        print(f'Accuracy on test data: {accuracy}')
        report(y_test, predicted)
        

elif model1 == 'CNN':
    model.eval()
    correct = 0
    total = 0
    y_pred = []
    y_test = []
    for x, y in test_dataloader:
        x = x.to(device)
        with torch.no_grad():
            yp = model(x)
        yp = torch.argmax(yp.cpu(), dim = 1)
        y_test.append(y)
        y_pred.append(yp)
        correct += (yp == y).sum()
        total += len(y)
    print(f"Accuracy on Test Data {(correct * 100 / total):.2f}")
    y_pred = np.concatenate(y_pred)
    y_test = np.concatenate(y_test)
    y_pred = np.reshape(y_pred, (-1))
    y_test = np.reshape(y_test, (-1))
    report(y_test, y_pred)

elif model1 == 'LSTM':
    correct = 0
    total = 0
    y_pred = []
    y_test = []
    for x, y in test_dataloader:
        x = x.to(device)
        with torch.no_grad():
            yp = model(x)
        yp = torch.argmax(yp.cpu(), dim = 1)
        y_test.append(y)
        y_pred.append(yp)
        correct += (yp == y).sum()
        total += len(y)
    print(f"Accuracy on Test Data {(correct * 100 / total):.2f}")
    y_pred = np.concatenate(y_pred)
    y_test = np.concatenate(y_test)
    y_pred = np.reshape(y_pred, (-1))
    y_test = np.reshape(y_test, (-1))
    report(y_test, y_pred)

else:
    print("Model Name should be DNN, CNN or LSTM")