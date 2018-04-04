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
import sys

clusters = int(sys.argv[1])

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
    train_classes.append(original_spks.index(spkid))

x_train = np.array(train_input_array)

kmeans = KMeans(n_clusters=clusters, random_state=0).fit(x_train)

spk_predictions = kmeans.predict(x_train)
for (fname, pred) in zip(train_fnames, spk_predictions):
     spk_file = '../features/spk_id_keras/' + fname + '.spk'
     f = open(spk_file, 'w')
     f.write(str(pred) + '\n')
     f.close()


print spk_predictions
print classification_report(train_classes,spk_predictions)
print recall_score(train_classes,spk_predictions,average='macro')
print accuracy_score(train_classes,spk_predictions)

for (a,b) in zip(train_classes,spk_predictions):
    print a,b
