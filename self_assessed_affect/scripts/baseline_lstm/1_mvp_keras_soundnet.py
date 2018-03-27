#from utils import *
import numpy as np
import random
from keras.utils import to_categorical

window = 5
num_classes = 3
input_dim = 512
hidden = 256


def get_max_len(arr):
   '''
   This takes a list of lists as input and returns the maximum length
   '''
   max_len = 0
   for a in arr:
     if len(a) > max_len:
          max_len = len(a)
   return max_len


# Process labels
labels_file = '/home3/srallaba/data/ComParE2018_SelfAssessedAffect/lab/ComParE2018_SelfAssessedAffect.tsv'
labels = {}
ids = ['l','m','h']
f = open(labels_file)
cnt = 0 
for line in f:
  if cnt == 0:
    cnt+= 1
  else:
    line = line.split('\n')[0].split()
    fname = line[0].split('.')[0]
    lbl = ids.index(line[1])
    labels[fname] = lbl
    

# Process the dev
print("Processing Dev")
f = open('files.devel')
devel_input_array = []
devel_output_array = []
for line in f:
    line = line.split('\n')[0]
    input_file = '../features/soundnet/val/' + line + '.npz'
    A = np.load(input_file)
    a = A['arr_0']
    inp = np.mean(a[4],axis=0)
    devel_input_array.append(inp)
    devel_output_array.append(labels[line])


x_dev = np.zeros( (len(devel_input_array), 1601, input_dim), 'float32')
y_dev = np.zeros( (len(devel_input_array), num_classes ), 'float32')

for i, (x,y) in enumerate(zip(devel_input_array, devel_output_array)):
   x_dev[i] = x
   y_dev[i] = to_categorical(y,num_classes)   

# Process the train
print("Processing Train")
f = open('files.train')
train_input_array = []
train_output_array = []
for line in f:
    line = line.split('\n')[0]
    input_file = '../features/soundnet/train/' + line + '.npz'
    A = np.load(input_file)
    a = A['arr_0']
    inp = np.mean(a[4],axis=0)

    train_input_array.append(inp)
    train_output_array.append(labels[line])

x_train = np.zeros( (len(train_input_array), 1601, input_dim), 'float32')
y_train = np.zeros( (len(train_input_array), num_classes ), 'float32')

for i, (x,y) in enumerate(zip(train_input_array, train_output_array)):
   x_train[i] = x
   y_train[i] = to_categorical(y,num_classes)




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



print input_dim
print x_train[0]
print y_train[0]

global model
model = Sequential()
model.add(LSTM(hidden, return_sequences=True, input_shape=(1601, input_dim)))
model.add(LSTM(hidden, return_sequences=True))
model.add(LSTM(hidden))
model.add(Dense(hidden, activation='selu')) 
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, batch_size=64, epochs=6, shuffle=True, validation_data=(x_dev,y_dev))


