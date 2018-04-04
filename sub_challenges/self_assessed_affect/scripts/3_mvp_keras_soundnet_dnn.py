#from utils import *
from sklearn import tree
import numpy as np
import random
from keras.utils import to_categorical
from sklearn.metrics import recall_score, classification_report, accuracy_score
from keras.callbacks import *
import pickle, logging
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import sys
from keras import regularizers
from keras.layers import LeakyReLU
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout


# Flags
spk_id_flag = 0 # This doesnt do anything really
normalize_flag= 0 # This doesnt do anything for now
add_lowseg0 = 1
regulate_lowseg = 1
limit = 1450
emph_flag = 1
text_embedding_flag = 0
spk_text_embedding_flag = 0
scaling_flag = 0

num_classes = 3

if spk_id_flag == 1:
   input_dim = 513
else:
   input_dim = 512  

hidden = 1024
input_dim = 512

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
    
if add_lowseg0 == 1:
  lowseg0_files = sorted(os.listdir('/home3/srallaba/challenges/compare2018/SoundNet-tensorflow/soundnet_feats_selfassessed_lowseg0'))
  for file in lowseg0_files:
    fname = file.split('.')[0]
    labels[fname] = 0   



# Process the dev
print("Processing Dev")
f = open('files.devel.copy')
devel_input_array = []
devel_output_array = []
for line in f:
    line = line.split('\n')[0]
    if emph_flag == 0:
       input_file = '../features/soundnet/val/' + line + '.npz'
    else:
       input_file = '../../SoundNet-tensorflow/soundnet_feats_SAA_emphed/' + line + '.npz'
    spk_file = '../features/spk_id_keras/' + line + '.spk'
    A = np.load(input_file)
    a = A['arr_0']
    b = np.loadtxt(spk_file)
    if spk_id_flag == 1:
        inp = np.concatenate(( np.array([int(b)]), np.mean(a[4],axis=0) ) )
    elif text_embedding_flag == 1:
        b = np.loadtxt('../features/text_embeddings/' + line + '.txt')
        inp = np.concatenate((np.array(b), np.mean(a[4],axis=0) ))
    elif spk_text_embedding_flag == 1:
        c = np.loadtxt(spk_file)
        b = np.loadtxt('../features/text_embeddings/' + line + '.txt')
        inp = np.concatenate(( np.array([int(c)]), np.array(b), np.mean(a[4],axis=0) ))

    else:  
        inp = np.mean(a[4],axis=0) 
        #inp = np.concatenate(( np.mean(a[4],axis=0), np.mean(a[2],axis=0) ))
        #inp = np.mean(a[2],axis=0)

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
count_m = 0
count_h = 0
count_l = 0
for line in f:
    line = line.split('\n')[0]
    lbl = labels[line]
    if emph_flag == 0:
       input_file = '../features/soundnet/train/' + line + '.npz'
    else: 
       input_file = '../../SoundNet-tensorflow/soundnet_feats_SAA_emphed/' + line + '.npz'
    spk_file = '../features/spk_id_keras/' + line + '.spk'
    A = np.load(input_file)
    a = A['arr_0']
    b = np.loadtxt(spk_file)
    if spk_id_flag == 1:
        inp = np.concatenate(( np.array([int(b)]), np.mean(a[4],axis=0) ) )
    elif text_embedding_flag == 1:
        b = np.loadtxt('../features/text_embeddings/' + line + '.txt')
        inp = np.concatenate(( b, np.mean(a[4],axis=0) ))
    elif spk_text_embedding_flag == 1:
        c = np.loadtxt(spk_file)
        b = np.loadtxt('../features/text_embeddings/' + line + '.txt')
        inp = np.concatenate(( np.array([int(c)]), np.array(b), np.mean(a[4],axis=0) ))

    else:  
        inp = np.mean(a[4],axis=0)
        #inp = np.concatenate(( np.mean(a[4],axis=0), np.mean(a[2],axis=0) ))
        #inp = np.mean(a[2],axis=0)


    train_input_array.append(inp)
    train_output_array.append(labels[line])

if add_lowseg0 == 1:
  lowseg0_files = sorted(os.listdir('/home3/srallaba/challenges/compare2018/SoundNet-tensorflow/soundnet_feats_selfassessed_lowseg0'))
  ctr = 0
  for input_file in lowseg0_files:
   ctr += 1
   if regulate_lowseg == 1 and ctr < limit:
     A = np.load('/home3/srallaba/challenges/compare2018/SoundNet-tensorflow/soundnet_feats_selfassessed_lowseg0/' + input_file)
     a = A['arr_0']
     inp = np.mean(a[4],axis=0)
     #inp = np.concatenate(( np.mean(a[4],axis=0), np.mean(a[2],axis=0) ))
     #inp = np.mean(a[2],axis=0)

     train_input_array.append(inp)
     train_output_array.append(0)


x_train = np.array(train_input_array)
y_train = to_categorical(train_output_array,num_classes)
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
   y_test_pred = model.predict(x_test)
   for i,y in enumerate(y_test_pred):
       y = np.argmax(y)
       f.write(str(i+1) + ' ' + str(y+1) + ':' + str(ids[y]) +  ' ' + str(y+1) + ':' + str(ids[y])  + '\n')
   f.close()
     
 
def get_challenge_uar(epoch):
   cmd = 'perl format_pred.pl /home3/srallaba/data/ComParE2018_SelfAssessedAffect/arff/ComParE2018_SelfAssessedAffect.ComParE.devel.arff  submission_' + str(epoch) + '.txt submission.arff 6375'
   print cmd
   os.system(cmd)

   cmd = 'perl score.pl /home3/srallaba/data/ComParE2018_SelfAssessedAffect/arff/ComParE2018_SelfAssessedAffect.ComParE.devel.arff submission.arff 6375'
   print cmd
   os.system(cmd)



# Normalize
if normalize_flag == 1:
   from sklearn.preprocessing import normalize
   x_train = normalize(x_train,axis=0)
   x_dev = normalize(x_dev,axis=0)


# Scaling
if scaling_flag == 1:
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler().fit(x_train)
   x_train = scaler.transform(x_train)
   x_dev = scaler.transform(x_dev)


# Reshape the data for CNN

# reshape train data
x_train_r = np.zeros((len(x_train), input_dim, 3))
x_train_r[:, :, 0] = x_train[:, :input_dim]
x_train_r[:, :, 1] = x_train[:, :input_dim]
x_train_r[:, :, 2] = x_train[:, :input_dim]

# reshape validation data
x_dev_r = np.zeros((len(x_dev), input_dim, 3))
x_dev_r[:, :, 0] = x_dev[:, :input_dim]
x_dev_r[:, :, 1] = x_dev[:, :input_dim]
x_dev_r[:, :, 2] = x_dev[:, :input_dim]



x_test = x_dev_r
y_train_r = y_train
y_dev_r = y_dev


print "I am causing the trouble", x_train.shape

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
from keras.layers import LSTM,Convolution1D
from keras.layers import Bidirectional
from keras.callbacks import *
import pickle, logging
from sklearn.metrics import confusion_matrix
from keras import optimizers
from keras import regularizers


global model
model = Sequential()

model.add(Convolution1D(nb_filter=32, filter_length=1, input_shape=(input_dim,3)))
model.add(LeakyReLU(alpha=0.3))
model.add(Flatten())
model.add(Dropout(0.4))

model.add(Dense(hidden))
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.2))

model.add(Dense(hidden, ))
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.2))


'''
model.add(Dense(hidden, activation='LeakyReLU', ))
model.add(Dropout(0.2))

model.add(Dense(hidden, activation='LeakyReLU', ))
model.add(Dropout(0.2))

model.add(Dense(hidden, activation='LeakyReLU', ))
model.add(Dropout(0.2))
'''

model.add(Dense(hidden, ))
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.2))


model.add(Dense(num_classes, activation='softmax'))
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.95, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
model.summary()
model.fit(x_train_r, y_train_r, epochs=12, batch_size=24, shuffle=True, validation_data=(x_dev_r,y_dev_r))



test(10)
get_challenge_uar(10)
