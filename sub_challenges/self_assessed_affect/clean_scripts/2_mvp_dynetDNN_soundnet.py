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
for line in f:
    line = line.split('\n')[0]
    input_file = '../features/soundnet/val/' + line + '.npz'
    A = np.load(input_file)
    a = A['arr_0']
    inp = np.mean(a[4],axis=0)
    devel_input_array.append(inp)
    devel_output_array.append(labels[line])

x_dev = np.array(devel_input_array)
y_dev = np.array(devel_output_array)

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

x_train = np.array(train_input_array)
y_train = np.array(train_output_array)

# Process the test

x_test = x_dev


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
   y_test_pred = []
   for k, (i,o) in enumerate(dev_data):
        dy.renew_cg()
        i = dy.inputVector(i)
        pred = dnn.predict(i)
        prediction = np.argmax(pred.value())
        y_test_pred.append(prediction)
        f.write(str(k) + ' ' + str(prediction) + ':' + str(ids[prediction]) +  ' ' + str(prediction) + ':' + str(ids[prediction])  + '\n')
   f.close()
   print classification_report(y_dev, y_test_pred)  
   print accuracy_score(y_dev, y_test_pred)
   print recall_score(y_dev,y_test_pred, average='macro')
 
def get_challenge_uar(epoch):
   cmd = 'perl format_pred.pl /home3/srallaba/data/ComParE2018_SelfAssessedAffect/arff/ComParE2018_SelfAssessedAffect.ComParE.devel.arff  submission_' + str(epoch) + '.txt submission.arff 6375'
   print cmd
   os.system(cmd)

   cmd = 'perl score.pl /home3/srallaba/data/ComParE2018_SelfAssessedAffect/arff/ComParE2018_SelfAssessedAffect.ComParE.devel.arff submission.arff 6375'
   print cmd
   os.system(cmd)


# Instantiate DNN and define the loss
m = dy.Model()
dnn = FeedForwardNeuralNet(m, [input_dim, [hidden, hidden], num_classes, [dy.selu, dy.selu, dy.selu]]) 
trainer = dy.CyclicalSGDTrainer(m)
update_params = 2
num_epochs = 10
num_embed = 3
lookup = m.add_lookup_parameters((num_classes, num_embed))

train_data = zip(x_train, y_train)
dev_data = zip(x_dev, y_dev)
num_train = len(train_data)
startTime = time.time()

# Loop over the training instances and call the mlp
for epoch in range(num_epochs):
  start_time = time.time()
  print " Epoch ", epoch
  train_loss = 0
  recons_loss = 0
  random.shuffle(train_data)
  K = 0
  count = 0
  frame_count = 0
  KL_loss = 0
  RECONS_loss = 0
  for (i,o) in train_data:
       count = 1
       K += 1
       dy.renew_cg()
       frame_count += 1
       count += 1
       i = dy.inputVector(i)
       #o = lookup[o]
       loss  = dnn.calculate_loss_classification(i, o)
       train_loss += loss.value()
       loss.backward()
       if frame_count % int(0.1*num_train) == 1:
           print "   Train Loss after processing " +  str(frame_count) + " number of files : " +  str(float(train_loss/frame_count))
       if frame_count % update_params == 1:
            trainer.update() 
  end_time = time.time()
  duration = end_time - start_time
  start_time = end_time
  print "Train Loss after epoch " +  str(epoch) + " : " +  str(float(train_loss/frame_count)), " with ", frame_count, " frames, in ", float((end_time - startTime)/60)  , " minutes "  
  print "I think I will run for another ", float( duration * ( num_epochs - epoch) / 60 ), " minutes "
  print '\n'

  test(epoch)
  #get_challenge_uar(epoch)
