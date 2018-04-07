import numpy as np
import random
from keras.utils import to_categorical
from sklearn.metrics import recall_score, classification_report, accuracy_score
from keras.callbacks import *
import pickle, logging
from keras.layers import Dense, Dropout,Bidirectional, TimeDistributed,AlphaDropout
from dynet_modules import *
import dynet as dy
import time
from sklearn.cluster import KMeans
import sys
import keras
from sklearn import preprocessing
import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense, AlphaDropout
from keras.callbacks import *
import pickle, logging
from keras import regularizers
import time, random
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.callbacks import *
import pickle, logging
from sklearn.metrics import confusion_matrix
from keras import optimizers
from keras import regularizers
from sklearn.model_selection import train_test_split

hidden = 128
input_dim = 512
num_classes = 10

original_spks = ['11', '03', '13', '12', '15', '14', '10', '08', '09', '16']

# Process the train
print("Processing Train")
files = sorted(os.listdir('../../SoundNet-tensorflow/soundnet_feats_emodb/'))
train_input_array = []
train_fnames = []
train_classes = []
for line in files:
    input_file = '../../SoundNet-tensorflow/soundnet_feats_emodb/' + line 
    A = np.load(input_file)
    a = A['arr_0']
    inp = np.mean(a[4],axis=0)
    train_fnames.append(line)
    train_input_array.append(inp)
    spkid = line[0:2]
    id = original_spks.index(spkid)
    train_classes.append(to_categorical(id,num_classes))

x_train = np.array(train_input_array)
y_train = np.array(train_classes)

input_scaler = preprocessing.StandardScaler().fit(x_train)
output_scaler = preprocessing.StandardScaler().fit(y_train)
x_train = input_scaler.transform(x_train)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.33, random_state=42)

global model
model = Sequential()
model.add(Dense(hidden, activation='selu',  input_shape=(input_dim,)))
model.add(AlphaDropout(0.3))
model.add(Dense(hidden, activation='selu',))
model.add(AlphaDropout(0.3))
model.add(Dense(hidden, activation='selu',))
model.add(AlphaDropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) 
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, epochs=600, batch_size=32, shuffle=True,validation_data=(x_test,y_test))


