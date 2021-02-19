import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
import re
import pickle
import csv
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


X_train_or = pd.read_csv('data/train.csv',names=["headline","y"])
X_valid_or = pd.read_csv('data/valid.csv',names=["headline","y"])
X_test_or = pd.read_csv('data/test.csv',names=["headline","y"])

stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess(data):

    new_data=[]
    for iter in range(len(data)):
        curText=data.iloc[iter,0]
        if len(curText) > 0:
            ascii_data = re.sub('[^A-Za-z0-9\']+', ' ', curText)
            text = [w for w in ascii_data.split() if w not in stop_words]
            lem_text = [lemmatizer.lemmatize(i) for i in text]
            stem_text = ""
            for abc in lem_text:
                # stem = "".join([stemmer.stem(i) for i in abc])
                stem = "".join([i for i in abc])
                stem_text = stem_text + " " + stem
            new_data.append(stem_text)
        else:
            new_data.append(np.NaN)
    data['headline']=new_data
      
    return data

# print(X_train_or.iloc[1,0])
X_train_pre = preprocess(X_train_or)
X_valid_pre = preprocess(X_valid_or)
X_test_pre = preprocess(X_test_or)
# X_train_pre = X_train_or
# X_valid_pre = X_valid_or
# X_test_pre = X_test_or
# print(X_train_pre.iloc[1,0])

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 64
# This is fixed.
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(X_train_pre['headline'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X_train = tokenizer.texts_to_sequences(X_train_pre['headline'].values) # Every word got a new number
# print(X_train[1])

X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of train data tensor:', X_train.shape)
# print(X_train[1])

X_test = tokenizer.texts_to_sequences(X_test_pre['headline'].values) # Every word got a new number
# print(X_test[1])

X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of test data tensor:', X_test.shape)
# print(X_test[1])

X_valid = tokenizer.texts_to_sequences(X_valid_pre['headline'].values) # Every word got a new number
# print(X_valid[1])

X_valid = pad_sequences(X_valid, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of valid data tensor:', X_valid.shape)
# print(X_valid[1])

Y_train = pd.get_dummies(X_train_pre['y']).values
print('Shape of train label tensor:', Y_train.shape)

Y_test = pd.get_dummies(X_test_pre['y']).values
print('Shape of test label tensor:', Y_test.shape)

Y_valid = pd.get_dummies(X_valid_pre['y']).values
print('Shape of valid label tensor:', Y_valid.shape)

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X_train.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(196, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

epochs = 2
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_valid, Y_valid),callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
name = "2_64"

np.save("history_"+name+".npy",history)
pickle.dump(model, open(name, 'wb'))

# Plot training & validation accuracy values
plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig("Accuracyvsepoch_"+name+".png")
# plt.show()

# Plot training & validation loss values
plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig("lossvsepoch_"+name+".png")
# plt.show()

# # Make predictions
y_pred = model.predict(X_test, batch_size=128, verbose=2)
print(type(y_pred))
print(y_pred.shape)
# print("Accuracy of test data is: ",accuracy(y_pred,Y_valid))

cm=confusion_matrix(np.argmax(Y_test, axis=1),np.argmax(y_pred, axis=1))
print("\n\nConfusion Matrix\n")
print(cm)

cr=classification_report(np.argmax(Y_test, axis=1),np.argmax(y_pred, axis=1))
print("\n\nClassification Report\n")
print(cr)

with open("predictions_"+name+".csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(y_pred)
