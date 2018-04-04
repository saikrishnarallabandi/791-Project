import numpy as np
import random
from keras.utils import to_categorical
from sklearn.metrics import recall_score, classification_report, accuracy_score
from keras.callbacks import *
import pickle, logging
from keras.layers import Dense, Dropout,Bidirectional, TimeDistributed
from dynet_modules import *
import dynet as dy
import time
from sklearn.cluster import KMeans

num_classes = 3
input_dim = 512
hidden =512


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
    if lbl > 0:
       lbl = 1
    labels[fname] = lbl
        

# Process the dev
print("Processing Dev")
f = open('files.devel.copy')
devel_input_array = []
devel_output_array = []
devel_fnames = []
for line in f:
    line = line.split('\n')[0]
    input_file = '../features/soundnet/val/' + line + '.npz'
    A = np.load(input_file)
    a = A['arr_0']
    inp = np.mean(a[4],axis=0)
    devel_input_array.append(inp)
    devel_output_array.append(labels[line])
    devel_fnames.append(line)

x_dev = np.array(devel_input_array)
y_dev = np.array(devel_output_array)

# Process the train
print("Processing Train")
f = open('files.train')
train_input_array = []
train_output_array = []
train_fnames = []
for line in f:
    line = line.split('\n')[0]
    input_file = '../features/soundnet/train/' + line + '.npz'
    A = np.load(input_file)
    a = A['arr_0']
    inp = np.mean(a[4],axis=0)
    train_fnames.append(line)
    train_input_array.append(inp)
    train_output_array.append(labels[line])

x_train = np.array(train_input_array)
y_train = np.array(train_output_array)

# Process the test
x_test = x_dev



kmeans = KMeans(n_clusters=100, random_state=0).fit(x_train)

spk_predictions = kmeans.predict(x_dev)
for (fname, pred) in zip(devel_fnames, spk_predictions):
     spk_file = '../features/spk_id_keras/' + fname + '.spk'
     f = open(spk_file, 'w')
     f.write(str(pred) + '\n')
     f.close()

spk_predictions = kmeans.predict(x_train)
for (fname, pred) in zip(train_fnames, spk_predictions):
     spk_file = '../features/spk_id_keras/' + fname + '.spk'
     f = open(spk_file, 'w')
     f.write(str(pred) + '\n')
     f.close()


