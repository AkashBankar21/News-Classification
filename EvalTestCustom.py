import pandas as pd
import re
import emoji
import fasttext
import numpy as np

# some NLP libraries that can help us with preprocessing
import nltk
nltk.download('stopwords')
nltk.download('punkt')

import emoji
import torch
import sys
import torch
from LSTM import CustomDataset, LSTMNetwork
from CNN import Network
from DNN import Model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, f1_score, recall_score
from torch.utils.data import DataLoader

def report(y_test, predicted):
    print("Classification Report for DNN :\n", classification_report(y_test, predicted))
    print("Precision : ", precision_score(y_test, predicted))
    print("Recall : ", recall_score(y_test, predicted))
    print("F1-Score : ", f1_score(y_test, predicted))
    print("Confusion Matrix : ", confusion_matrix(y_test, predicted))


# Check if arguments are provided
if len(sys.argv) < 4:
    print("Usage: python script.py <path to data> <path to model> <model type>")
    sys.exit(1)

# Access command-line arguments
data_path = str(sys.argv[1])
model_path = str(sys.argv[2])
model1 = sys.argv[3]

data = pd.read_csv(data_path)

def convert_to_emoji(text):
    return emoji.demojize(text)

tweets = data['tweet']
tweets_tokenized = []
tt = nltk.tokenize.TweetTokenizer()
for tweet in tweets:
    tweet = convert_to_emoji(tweet)
    # initializing punctuations string
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~"'?'''
 
    # Removing punctuations in string
    # Using loop + punctuation string
    for ele in tweet:
        if ele in punc:
            tweet = tweet.replace(ele, "")
 
    tweets_tokenized.append(tt.tokenize(text=tweet))

#remove stopwords
stemmer = nltk.stem.porter.PorterStemmer()
stoplist = nltk.corpus.stopwords.words('english')

stemmed = []

for tweet in tweets_tokenized:
    processed_tokens = [stemmer.stem(t) for t in tweet if t not in stoplist]
    stemmed.append(processed_tokens)

def super_simple_preprocess(text):
  # lowercase
  text = text.lower()
  # remove non alphanumeric characters
  text = re.sub('[^A-Za-z0-9 ]+','', text)
  return text

data_updated = []
for tweet in stemmed:
    data_updated.append(super_simple_preprocess(" ".join(tweet)))
pd.DataFrame(data_updated).shape

data_updated = pd.DataFrame(data_updated)
data_updated.insert(1, 'w', data['label'])
data = data_updated

sentences = data.iloc[:, 0]
labels = data['w']

labels.replace('real',1, inplace=True)
labels.replace('fake',0, inplace=True)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)
X_test = X.toarray()
y_test = labels

X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test.values)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load(model_path)

def preprocessforcnn():
    sentences = data.iloc[:, 0]
    labels = data['w']
    # Load the pre-trained FastText model
    fasttext_model = fasttext.load_model('./model.bin')
    # Tokenize input sentences
    tokenized_sentences = [sentence.split() for sentence in sentences]
    # Determine the maximum input length
    max_inp_len = max(len(tokens) for tokens in tokenized_sentences)
    # Dimension of FastText embeddings
    d = fasttext_model.get_dimension()
    # Initialize matrix with zeros
    matrix = np.zeros((len(sentences), max_inp_len, d), dtype=np.float32)
    # Fill the matrix with FastText embeddings
    for i, tokens in enumerate(tokenized_sentences):
        for j, token in enumerate(tokens):
            matrix[i, j] = fasttext_model[token]

    test_matrix = matrix
    test_labels = labels

    test_matrix = pd.DataFrame(test_matrix.reshape(-1, 106400))
    test_labels = pd.DataFrame(test_labels)
    return test_matrix, test_labels

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
    test_matrix, test_labels = preprocessforcnn()
    test_dataset = CustomDataset(test_matrix, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    model.eval()
    correct = 0
    total = 0
    for x, y in test_dataloader:
        x = x.to(device)
        with torch.no_grad():
            yp = model(x)
        yp = torch.argmax(yp.cpu(), dim = 1)
        correct += (yp == y).sum()
        total += len(y)
    print(f"Accuracy on Test Data {(correct * 100 / total):.2f}")
    report(y, yp)

elif model1 == 'LSTM':
    test_matrix, test_labels = preprocessforcnn()
    test_dataset = CustomDataset(test_matrix, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    model.eval()
    correct = 0
    total = 0
    for x, y in test_dataloader:
        x = x.to(device)
        with torch.no_grad():
            yp = model(x)
        yp = torch.argmax(yp.cpu(), dim = 1)
        correct += (yp == y).sum()
        total += len(y)
    print(f"Accuracy on Val Data {(correct * 100 / total):.2f}")
    
    report(y, yp)

else:
    print("Model Name should be DNN, CNN or LSTM")