import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import emoji

# some NLP libraries that can help us with preprocessing
import nltk
nltk.download('stopwords')
nltk.download('punkt')

import preprocessor as tweet_preprocessor
from ttp import ttp
import emoji
import torch
import fasttext
import pickle

data = pd.read_csv('./data.csv')

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
X = X.toarray()
y = labels

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)
print("Train : ", len(X_train),"Test : ", len(X_test),"Val : ", len(X_val))

X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train.values)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test.values)
X_val = torch.FloatTensor(X_val)
y_val = torch.LongTensor(y_val.values)

with open('X_train.pickle', 'wb') as file:
    pickle.dump(X_train, file)
with open('y_train.pickle', 'wb') as file:
    pickle.dump(y_train, file)
with open('X_test.pickle', 'wb') as file:
    pickle.dump(X_test, file)
with open('y_test.pickle', 'wb') as file:
    pickle.dump(y_test, file)
with open('X_val.pickle', 'wb') as file:
    pickle.dump(X_val, file)
with open('y_val.pickle', 'wb') as file:
    pickle.dump(y_val, file)
 
data = pd.read_csv('./data.csv')
labels = data['label']
labels.replace('real',1, inplace=True)
labels.replace('fake',0, inplace=True)
data['combined'] = "__label__" + data['label'].astype(str) + " " + data['tweet'].astype(str)
train, test = train_test_split(data, test_size=0.2)
test, val = train_test_split(test, test_size=0.5)

train.to_csv('data.train', columns=['combined'], index=False, header=False)
test.to_csv('data.test', columns=['combined'], index=False, header=False)
val.to_csv('data.val', columns=['combined'], index=False, header=False)

model = fasttext.train_supervised(input='data.train')
model.test('data.test')

model.save_model(path='./model.bin')

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

train_matrix = matrix[:8480, :, :]
test_matrix = matrix[9540:, :, :]
val_matrix = matrix[8480 : 9540, :, :]
train_labels = labels[:8480]
test_labels = labels[9540:]
val_labels = labels[8480 : 9540]

train_matrix = pd.DataFrame(train_matrix.reshape(-1,106400))
test_matrix = pd.DataFrame(test_matrix.reshape(-1, 106400))
val_matrix = pd.DataFrame(val_matrix.reshape(-1, 106400))
train_labels = pd.DataFrame(train_labels)
test_labels = pd.DataFrame(test_labels)
val_labels = pd.DataFrame(val_labels)

with open('train_matrix.pickle', 'wb') as file:
    pickle.dump(train_matrix, file)
with open('test_matrix.pickle', 'wb') as file:
    pickle.dump(test_matrix, file)
with open('val_matrix.pickle', 'wb') as file:
    pickle.dump(val_matrix, file)
with open('train_labels.pickle', 'wb') as file:
    pickle.dump(train_labels, file)
with open('test_labels.pickle', 'wb') as file:
    pickle.dump(test_labels, file)
with open('val_labels.pickle', 'wb') as file:
    pickle.dump(val_labels, file)


print("Preprocessing Finished!")