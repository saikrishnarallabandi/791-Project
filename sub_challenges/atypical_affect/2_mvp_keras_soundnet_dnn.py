#from utils import *
import numpy as np
import random
from keras.utils import to_categorical
from sklearn.metrics import recall_score, classification_report
from keras.callbacks import *
import pickle, logging
from sklearn import preprocessing

window = 5
num_classes = 4
input_dim = 512
hidden = 512


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
labels_file = '/home3/srallaba/challenges/compare2018/atypical_affect/ComParE2018_AtypicalAffect_v1.1/lab/ComParE2018_AtypicalAffect.tsv'
labels = {}
ids = ['happy','neutral','angry', 'sad']
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
    input_file = '/home3/srallaba/challenges/compare2018/SoundNet-tensorflow/soundnet_feats_atypical//' + line + '.npz'
    A = np.load(input_file)
    a = A['arr_0']
    inp_5 = np.mean(a[2],axis=0)
    inp_7 = np.mean(a[4],axis=0)
    inp = np.concatenate((inp_5,inp_7))
    inp = np.mean(a[4],axis=0)
    devel_input_array.append(inp)
    devel_output_array.append(labels[line])


x_dev = np.array(devel_input_array)
y_dev = to_categorical(devel_output_array,num_classes)
y_dev = np.array(y_dev)


# Process the train
print("Processing Train")
f = open('files.train.full')
train_input_array = []
train_output_array = []
for line in f:
    line = line.split('\n')[0]
    input_file = '/home3/srallaba/challenges/compare2018/SoundNet-tensorflow/soundnet_feats_atypical/' + line + '.npz'
    A = np.load(input_file)
    a = A['arr_0']
    inp_5 = np.mean(a[2],axis=0)
    inp_7 = np.mean(a[4],axis=0)
    inp = np.concatenate((inp_5,inp_7))
    inp = np.mean(a[4],axis=0)
    train_input_array.append(inp)
    train_output_array.append(labels[line])

print "happy neutral angry sad"
print "Counts are: ", np.bincount(train_output_array)

x_train = np.zeros( (len(train_input_array), 1601, input_dim), 'float32')
y_train = np.zeros( (len(train_input_array), num_classes ), 'float32')

x_train = np.array(train_input_array)
y_train = to_categorical(train_output_array,num_classes)
#print "Counts are: ", np.bincount(y_train)

y_train = np.array(y_train)


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
   cmd = 'perl format_pred.pl /home3/srallaba/challenges/compare2018/atypical_affect/ComParE2018_AtypicalAffect_v1.1/arff/ComParE2018_AtypicalAffect.ComParE.devel.arff  submission_' + str(epoch) + '.txt submission.arff 6375'
   print cmd
   os.system(cmd)

   cmd = 'perl score.pl /home3/srallaba/challenges/compare2018/atypical_affect/ComParE2018_AtypicalAffect_v1.1/arff/ComParE2018_AtypicalAffect.ComParE.devel.arff submission.arff 6375'
   print cmd
   os.system(cmd)

class LoggingCallback(Callback):
    """Callback that logs message at end of epoch.
    """
    def __init__(self, print_fcn="print"):
        Callback.__init__(self)
        self.print_fcn = print_fcn

    def on_epoch_end(self, epoch, logs={}):
       pass
       #get_uar(epoch)
       test(epoch)
       get_challenge_uar(epoch)

x_test = x_dev

'''
input_scaler = preprocessing.StandardScaler().fit(x_train)
output_scaler = preprocessing.StandardScaler().fit(y_train)
train_input = input_scaler.transform(x_train)
#train_output = output_scaler.transform(y_train)
train_output = y_train
dev_input = input_scaler.transform(x_dev)
x_test = x_dev

train_input = x_train
dev_input = x_dev
'''




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


global model
model = Sequential()

model.add(Dense(hidden, activation='relu',  input_shape=(input_dim,)))
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
model.fit(x_train, y_train, epochs=15, batch_size=32, shuffle=True, validation_data=(x_dev,y_dev))


test(15)
get_challenge_uar(15)

