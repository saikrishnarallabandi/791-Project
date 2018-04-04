#from utils import *
import numpy as np
import random
from keras.utils import to_categorical
from sklearn.metrics import recall_score, classification_report
from keras.callbacks import *
import pickle, logging
from keras.layers import Dense, Dropout,Bidirectional, TimeDistributed

window = 5
num_classes = 3
input_dim = 50
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
    
binary2id = {i:w for w,i in labels.iteritems()}


# Process the dev
print("Processing Dev")
f = open('files.devel.copy')
devel_input_array = []
devel_output_array = []
for line in f:
    line = line.split('\n')[0]
    input_file = '../features/mcep_ascii/' + line + '.mcep'
    inp = np.loadtxt(input_file)
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
    input_file = '../features/mcep_ascii/' + line + '.mcep'
    inp = np.loadtxt(input_file)

    train_input_array.append(inp)
    train_output_array.append(labels[line])

x_train = np.zeros( (len(train_input_array), 1601, input_dim), 'float32')
y_train = np.zeros( (len(train_input_array), num_classes ), 'float32')

for i, (x,y) in enumerate(zip(train_input_array, train_output_array)):
   x_train[i] = x
   y_train[i] = to_categorical(y,num_classes)


# Process the test
print("Processing Test")
f = open('files.devel.copy')
test_input_array = []
for line in f:
    line = line.split('\n')[0]
    input_file = '../features/mcep_ascii/' + line + '.mcep'
    inp = np.loadtxt(input_file)
    test_input_array.append(inp)

x_test = np.zeros( (len(test_input_array), 1601, input_dim), 'float32')

for i, x in enumerate(test_input_array):
   x_test[i] = x



def get_uar(epoch):
   y_dev_pred_binary = model.predict(x_dev)
   y_dev_pred = []
   for y in y_dev_pred_binary:
       y_dev_pred.append(np.argmax(y))

   y_dev_ascii = []
   for y in y_dev:
       y_dev_ascii.append(np.argmax(y))

   print "UAR after epoch ", epoch, " is ", classification_report(y_dev_ascii, y_dev_pred)


def test(epoch):
   f = open('submission_' + str(epoch) + '.txt','w')
   f.write('inst# actual predicted' + '\n')
   y_test_pred_binary = model.predict(x_test)
   y_test_pred = []
   for i, y in enumerate(y_test_pred_binary):
       y_test_pred.append(np.argmax(y))
       prediction = np.argmax(y) 
       f.write(str(i) + ' ' + str(prediction) + ':' + str(ids[prediction]) +  ' ' + str(prediction) + ':' + str(ids[prediction])  + '\n')
   f.close()
     
 
def get_challenge_uar(epoch):
   cmd = 'perl format_pred.pl /home3/srallaba/data/ComParE2018_SelfAssessedAffect/arff/ComParE2018_SelfAssessedAffect.ComParE.devel.arff  submission_' + str(epoch) + '.txt submission.arff 6375'
   print cmd
   os.system(cmd)

   cmd = 'perl score.pl /home3/srallaba/data/ComParE2018_SelfAssessedAffect/arff/ComParE2018_SelfAssessedAffect.ComParE.devel.arff submission.arff 6375'
   print cmd
   os.system(cmd)

class LoggingCallback(Callback):
    """Callback that logs message at end of epoch.
    """
    def __init__(self, print_fcn="print"):
        Callback.__init__(self)
        self.print_fcn = print_fcn

    def on_epoch_end(self, epoch, logs={}):
 
       #get_uar(epoch)
       test(epoch)
       get_challenge_uar(epoch)

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


global model
model = Sequential()
model.add(Bidirectional(LSTM(hidden, return_sequences=True), input_shape=(1601, input_dim)))
model.add(Bidirectional(LSTM(hidden)))
model.add(Dense(hidden, activation='selu')) 
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, epochs=10, shuffle=True, validation_data=(x_dev,y_dev), callbacks=[LoggingCallback(logging.info)])


