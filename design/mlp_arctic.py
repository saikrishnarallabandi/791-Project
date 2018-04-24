import numpy as np
import random
from keras.utils import to_categorical
from sklearn.metrics import recall_score, classification_report
from keras.callbacks import *
import pickle, logging
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import sys
from keras.models import Sequential
from keras.layers import Dense, AlphaDropout, Dropout
from keras import regularizers
import time, random
from keras.layers import Embedding
from sklearn.metrics import confusion_matrix
from keras import optimizers
from keras import regularizers



          
num_classes = 12
input_dim = 128
hidden = 1024 
        

sound_path = "/home2/compare791/challenges/modules/spk_ID/tools/SoundNet-tensorflow/soundnet_feats_arctic/"

ids = ['awb','bdl','fem','gka','clb','aup','ahw','rxr', 'slt','ksp','jmk','rms' ]
    
    

f = open('list_files.txt')
train_input_array = []
train_output_array = []
for line in f:
    line = line.strip('\r\n').split('/')[7]
    input_file=sound_path+line.split('.')[0] + '.npz'
    A = np.load(input_file)
    a = A['arr_0']
    inp = np.mean(a[2],axis=0)
    train_input_array.append(inp)
    out = line.split('_')[0].split('-')[0]
    train_output_array.append(ids.index(out))

print train_output_array

combined = list(zip(train_input_array, train_output_array))
random.shuffle(combined)

train_input_array[:], train_output_array[:] = zip(*combined)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(train_input_array, train_output_array, test_size=0.20, random_state=5)



x_train = np.array(Xtrain)
y_train = to_categorical(Ytrain,num_classes)
y_train = np.array(y_train)


x_dev = np.array(Xtest)
y_dev = to_categorical(Ytest,num_classes)
y_dev = np.array(y_dev)

global model
model = Sequential()

model.add(Dense(hidden, activation='selu',  input_shape=(input_dim,)))
model.add(Dropout(0.2))

model.add(Dense(hidden, activation='relu',))
model.add(Dropout(0.2))
model.add(Dense(hidden, activation='relu',))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
model.summary()

filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(x_train, y_train, epochs=40, batch_size=32, shuffle=True, validation_data=(x_dev,y_dev),callbacks=callbacks_list)                                                                                                                                                                                    38,1          72%
