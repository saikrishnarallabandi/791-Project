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
import random


spk_id_flag = 0
normalize_flag= 0
num_classes = 3
add_lowseg0 = 1
regulate_lowseg = 1
limit = 1200
emph_flag = 0
text_embedding_flag = 0
spk_text_embedding_flag = 0
scaling_flag = 0
add_mediumsegments = 1
regulate_medseg = 1
limit_medium = 2000
add_highsegments = 1
regulate_highseg = 1
limit_high = 2000
testing_flag = 1
layer = 4


if spk_id_flag:
   input_dim = 513
else:
   input_dim = 512
hidden = 1024

m_limit=int(float(sys.argv[1]) * 388)
h_limit=int(float(sys.argv[2]) * 363)

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

if add_mediumsegments == 1:
  medium_files = sorted(os.listdir('/home3/srallaba/challenges/compare2018/SoundNet-tensorflow/soundnet_feats_medium_segments'))
  for file in medium_files:
    fname = file.split('.')[0]
    labels[fname] = 1

if add_highsegments == 1:
  high_files = sorted(os.listdir('/home3/srallaba/challenges/compare2018/SoundNet-tensorflow/soundnet_feats_high_segments'))
  for file in high_files:
    fname = file.split('.')[0]
    labels[fname] = 1


# Process the dev
print("Processing Dev")
f = open('files.devel.copy')
devel_input_array = []
devel_output_array = []
for line in f:
    line = line.split('\n')[0]
    if emph_flag == 1:
       input_file = '../../SoundNet-tensorflow/soundnet_feats_SAA_emphed/' + line + '.npz'
    else:
       input_file = '../features/soundnet/val/' + line + '.npz'
    spk_file = '../features/spk_id_keras/' + line + '.spk'
    A = np.load(input_file)
    a = A['arr_0']
    b = np.loadtxt(spk_file)
    if spk_id_flag == 1:
        inp = np.concatenate(( np.array([int(b)]), np.mean(a[layer],axis=0) ) )
    else:
        inp = np.mean(a[layer],axis=0)
    devel_input_array.append(inp)
    devel_output_array.append(labels[line])



x_dev = np.array(devel_input_array)
y_dev = np.array(devel_output_array)


# Process the test
print("Processing Test")
f = open('files.test')
test_input_array = []
for line in f:
    line = line.split('\n')[0]
    if emph_flag == 1:
       input_file = '../../SoundNet-tensorflow/soundnet_feats_SAA_emphed/' + line + '.npz'
    else:
       input_file = '../features/soundnet/test/' + line + '.npz'
    spk_file = '../features/spk_id_keras/' + line + '.spk'
    A = np.load(input_file)
    a = A['arr_0']
    if spk_id_flag == 1:
        b = np.loadtxt(spk_file)
        inp = np.concatenate(( np.array([int(b)]), np.mean(a[layer],axis=0) ) )
    else:
        inp = np.mean(a[layer],axis=0)
    test_input_array.append(inp)


x_test = np.array(test_input_array)





# Process the train
print("Processing Train")
f = open('files.train.full')
lines = f.readlines()
random.shuffle(lines)
train_input_array = []
train_output_array = []
count_m = 0
count_h = 0
count_l = 0
for line in lines:
    line = line.split('\n')[0]
    lbl = labels[line]
    if lbl == 1 and count_m > m_limit:
       continue
    elif lbl == 2 and count_h > h_limit:
       continue
    else:
      pass
    if emph_flag == 1:
       input_file = '../../SoundNet-tensorflow/soundnet_feats_SAA_emphed/' + line + '.npz'
    else:
       input_file = '../features/soundnet/train/' + line + '.npz'
    spk_file = '../features/spk_id_keras/' + line + '.spk'
    A = np.load(input_file)
    a = A['arr_0']
    b = np.loadtxt(spk_file)
    if spk_id_flag == 1:
        inp = np.concatenate(( np.array([int(b)]), np.mean(a[layer],axis=0) ) )
    else:
        inp = np.mean(a[layer],axis=0)
    train_input_array.append(inp)
    train_output_array.append(labels[line])
    if lbl == 1:
        count_m += 1
    elif lbl == 2:
        count_h += 1


if add_lowseg0 == 1:
  lowseg0_files = sorted(os.listdir('/home3/srallaba/challenges/compare2018/SoundNet-tensorflow/soundnet_feats_selfassessed_lowseg0'))
  ctr = 0
  for input_file in lowseg0_files:
   ctr += 1
   if regulate_lowseg == 1 and ctr < limit:
     A = np.load('/home3/srallaba/challenges/compare2018/SoundNet-tensorflow/soundnet_feats_selfassessed_lowseg0/' + input_file)
     a = A['arr_0']
     if spk_id_flag == 1:
        spk_file = '../features/spk_id_keras/' + input_file.split('_seg')[0] + '.spk'
        inp = np.concatenate(( np.array([int(b)]), np.mean(a[layer],axis=0) ) )
     else:
        inp = np.mean(a[layer],axis=0)
     #inp = np.concatenate(( np.mean(a[4],axis=0), np.mean(a[2],axis=0) ))
     #inp = np.mean(a[2],axis=0)

     train_input_array.append(inp)
     train_output_array.append(0)


if add_mediumsegments == 1:
  medium_files = os.listdir('/home3/srallaba/challenges/compare2018/SoundNet-tensorflow/soundnet_feats_medium_segments')
  ctr = 0
  for input_file in medium_files:
   ctr += 1
   if regulate_medseg == 1 and ctr < limit_medium:
     A = np.load('/home3/srallaba/challenges/compare2018/SoundNet-tensorflow/soundnet_feats_medium_segments/' + input_file)
     a = A['arr_0']
     if spk_id_flag == 1:
        spk_file = '../features/spk_id_keras/' + input_file.split('_seg')[0] + '.spk'
        inp = np.concatenate(( np.array([int(b)]), np.mean(a[layer],axis=0) ) )
     else:
        inp = np.mean(a[layer],axis=0)
     #inp = np.concatenate(( np.mean(a[4],axis=0), np.mean(a[2],axis=0) ))
     #inp = np.mean(a[2],axis=0)
     train_input_array.append(inp)
     train_output_array.append(1)

if add_highsegments == 1:
  high_files = os.listdir('/home3/srallaba/challenges/compare2018/SoundNet-tensorflow/soundnet_feats_high_segments')
  ctr = 0
  for input_file in high_files:
   ctr += 1
   if regulate_highseg == 1 and ctr < limit_high:
     A = np.load('/home3/srallaba/challenges/compare2018/SoundNet-tensorflow/soundnet_feats_high_segments/' + input_file)
     a = A['arr_0']
     if spk_id_flag == 1:
        spk_file = '../features/spk_id_keras/' + input_file.split('_seg')[0] + '.spk'
        inp = np.concatenate(( np.array([int(b)]), np.mean(a[layer],axis=0) ) )
     else:
        inp = np.mean(a[layer],axis=0)
     #inp = np.concatenate(( np.mean(a[4],axis=0), np.mean(a[2],axis=0) ))
     #inp = np.mean(a[2],axis=0)
     train_input_array.append(inp)
     train_output_array.append(2)


x_train = np.array(train_input_array)
y_train = np.array(train_output_array)
print "Counts are: ", np.bincount(y_train)

from sklearn.model_selection import KFold

def get_uar(epoch):
   
   y_dev_pred = clf.predict(x_dev)

   print "UAR after epoch ", epoch, " is ", classification_report(y_dev, y_dev_pred)

   print "I dont believe you. Lets do k fold"

   kf = KFold(n_splits=2)
   for test_index1, test_index2 in kf.split(x_dev):
       X_test1, X_test2 = x_dev[test_index1], x_dev[test_index2]
       y_test1, y_test2 = y_dev[test_index1], y_dev[test_index2]

       ypred1 = clf.predict(X_test1)
       print "UAR is ", classification_report(y_test1, ypred1)
   
       ypred2 = clf.predict(X_test2)
       print "UAR is ", classification_report(y_test2, ypred2)

 

def test(epoch):
   f = open('submission_' + str(epoch) + '.txt','w')
   f.write('inst# actual predicted' + '\n')
   y_test_pred = clf.predict(x_test)
   for i,y in enumerate(y_test_pred):
       f.write(str(i+1) + ' ' + str(y+1) + ':' + str(ids[y]) +  ' ' + str(y+1) + ':' + str(ids[y])  + '\n')
   f.close()
     
 
def get_challenge_uar(epoch):
   cmd = 'perl format_pred.pl /home3/srallaba/data/ComParE2018_SelfAssessedAffect/arff/ComParE2018_SelfAssessedAffect.ComParE.devel.arff  submission_' + str(epoch) + '.txt submission.arff 6375'
   print cmd
   os.system(cmd)

   cmd = 'perl score.pl /home3/srallaba/data/ComParE2018_SelfAssessedAffect/arff/ComParE2018_SelfAssessedAffect.ComParE.devel.arff submission.arff 6375'
   print cmd
   os.system(cmd)


if testing_flag == 0:
   x_test = x_dev



# Normalize
if normalize_flag ==1 :
   from sklearn.preprocessing import normalize
   x_train = normalize(x_train,axis=0)
   x_dev = normalize(x_dev,axis=0)



'''
print "Running SVM"
clf = SVC(kernel='rbf') #,class_weight='balanced')
clf = clf.fit(x_train, y_train)
y_dev_pred = clf.predict(x_dev)
test(0)
get_challenge_uar(0)
'''

print "Running Random Forest"
clf = RandomForestClassifier(n_estimators=10) #,conf=[0.95,0.95,0.95])
clf = clf.fit(x_train, y_train)
if testing_flag == 1:
   y_test = clf.predict(x_test)
   g = open('test_predictions.txt','w')
   for i,y in enumerate(y_test):
       g.write("'test_" + str(i+1).zfill(4) + ".wav'," + str(ids[y]) + '\n')
   g.close()

if testing_flag == 0:
   test(0)
   get_challenge_uar(0)

#for k in range(100):
#   get_uar(0)

x_test = x_dev
test(0)
get_challenge_uar(0)

