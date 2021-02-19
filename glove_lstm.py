import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.utils.np_utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Dropout
import re
import pickle
import csv
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from keras import regularizers
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from sklearn.naive_bayes import MultinomialNB

# dim - 200 , stemming removed, trainable true   82 accuracy, 20 epochs, callback plateau min lr 10e-5, MAX_SEQUENCE_LENGTH = 32

BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'glove')

# first, build index mapping words in the embeddings set
# to their embedding vector


# print(os.path)
# print(GLOVE_DIR)
# print('Indexing word vectors.')
# embeddings_index = {}
# with open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt')) as f: #dimension is 200 , try 300 next
#     for line in f:
#         word, coefs = line.split(maxsplit=1)
#         coefs = np.fromstring(coefs, 'f', sep=' ')
#         embeddings_index[word] = coefs

# print('Found %s word vectors.' % len(embeddings_index))
# print(embeddings_index['if'])



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
            #print("ascii_data ", ascii_data)
            text = [w for w in ascii_data.split() if w not in stop_words]
            #print("text ", text)
            lem_text = [lemmatizer.lemmatize(i) for i in text]
            #print("lem_text ", lem_text)
            stem_text = ""
            for abc in lem_text:
                #stem = "".join([stemmer.stem(i) for i in abc])
                stem = "".join([i for i in abc])
                stem_text = stem_text + " " + stem
            new_data.append(stem_text) #stemming not used!
            if len(stem_text.split(" ")) > 100:
                print(stem_text)
            #print("stem_text ", stem_text)
        else:
            new_data.append(np.NaN)
    data['headline']=new_data
    return data


X_train_pre = preprocess(X_train_or)
X_valid_pre = preprocess(X_valid_or)
X_test_pre = preprocess(X_test_or)


MAX_SEQUENCE_LENGTH = 32
MAX_NUM_WORDS = 20000 #max token 19361 something
EMBEDDING_DIM = 300

print("MAXIMUM")
print(max([len(X_train_pre["headline"][i].strip().split(" ")) for i in range(len(X_train_pre))]))
print(max([len(X_valid_pre["headline"][i].strip().split(" ")) for i in range(len(X_valid_pre))]))
print(max([len(X_test_pre["headline"][i].strip().split(" ")) for i in range(len(X_test_pre))]))

#print(X_test_pre["headline"][2])


# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(X_train_pre['headline'].values)
X_train = tokenizer.texts_to_sequences(X_train_pre['headline'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of train data tensor:', X_train.shape)
#print("X_train[0] &&&&&&&&&&&&&&&&&&&&&&&& :",X_train[0])
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


print('Preparing embedding matrix.')
# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index) + 1)




# count=0
# embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
# for word, i in word_index.items():
#     if i >= MAX_NUM_WORDS:
#         continue
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         # words not found in embedding index will be all-zeros.
#         embedding_matrix[i] = embedding_vector
#     else:
#         count+=1
# np.save("embedding_matrix_300.npy",embedding_matrix)


# 

#embedding_matrix=np.load("embedding_matrix.npy") # for 200
embedding_matrix=np.load("embedding_matrix_300.npy")


#print("Missing words:" , count)
print(embedding_matrix[0])
# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

model = Sequential()
model.add(embedding_layer)
model.add(SpatialDropout1D(0.4)) #try changing this
#Take screenshot of models
acti='relu' #sigmoid
drop=0.2 #0,0.5
model.add(LSTM(196, dropout=drop , recurrent_dropout=0.2, activation=acti)) #increase units and increase dropout
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

epochs = 20
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_valid, Y_valid), callbacks=[ReduceLROnPlateau(monitor='val_loss', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.3, 
                                            min_lr=0.00001)])
name = "_"+str(epochs)+"_"+str(batch_size)+"_"+str(EMBEDDING_DIM)+"_"+str(drop)+"_"+str(acti)
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
plt.show()

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


#print("Accuracy of test data is: ",accuracy(y_pred,Y_valid))

y_pred = model.predict(X_valid, batch_size=128, verbose=2)
cm=confusion_matrix(np.argmax(Y_valid, axis=1),np.argmax(y_pred, axis=1))
print("\n\nConfusion Matrix\n")
print(cm)

cr=classification_report(np.argmax(Y_valid, axis=1),np.argmax(y_pred, axis=1))
print("\n\nClassification Report\n")
print(cr)


# # Make predictions
y_pred = model.predict(X_test, batch_size=128, verbose=2)
print(type(y_pred))
print(y_pred.shape)
cm=confusion_matrix(np.argmax(Y_test, axis=1),np.argmax(y_pred, axis=1))
print("\n\nConfusion Matrix\n")
print(cm)

cr=classification_report(np.argmax(Y_test, axis=1),np.argmax(y_pred, axis=1))
print("\n\nClassification Report\n")
print(cr)


print("Headlines: ")
for i in range(0,4):
    print(X_test_pre["headline"][i]," ", X_test_pre["y"][i], " ",y_pred[i])



with open("predictions_"+name+".csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(y_pred)

print(name)


# def MultinomialNaiveBayes():
#     clf = MultinomialNB()
#     clf.fit(X_train, X_train_pre['y'])
#     y_pred = clf.predict(X_test)
#     cm=confusion_matrix(np.argmax(Y_test, axis=1),y_pred)
#     cr=classification_report(np.argmax(Y_test, axis=1),y_pred)

# MultinomialNaiveBayes()

