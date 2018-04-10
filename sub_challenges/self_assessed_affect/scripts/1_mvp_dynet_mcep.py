#from utils import *
import sys
sys.path.append("absolute path of logs directory")
import log_config as l
import numpy as np
import random
from keras.utils import to_categorical
from sklearn.metrics import recall_score, classification_report
from keras.callbacks import *
import pickle, logging
from keras.layers import Dense, Dropout,Bidirectional, TimeDistributed
from dynet_modules import *
import dynet as dy
import time


window = 5
num_classes = 3
input_dim = 5
hidden = 256
l.set_exp_name("1_mvp_dynet_mcep")

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
l.logger("Processing Dev", 'INFO')
f = open('files.devel.copy')
devel_input_array = []
devel_output_array = []
for line in f:
    line = line.split('\n')[0]
    input_file = '../features/str/' + line + '.str'
    inp = np.loadtxt(input_file)
    devel_input_array.append(inp)
    devel_output_array.append(labels[line])

x_dev = np.array(devel_input_array)
y_dev = np.array(devel_output_array)

# Process the train
l.logger("Processing Train",'INFO')
f = open('files.train')
train_input_array = []
train_output_array = []
for line in f:
    line = line.split('\n')[0]
    input_file = '../features/str/' + line + '.str'
    inp = np.loadtxt(input_file)

    train_input_array.append(inp)
    train_output_array.append(labels[line])


x_train = np.array(train_input_array)
y_train = np.array(train_output_array)
l.logger( "Y train is: ")
l.logger(y_train)

# Process the test
l.logger("Processing Test", 'INFO')
f = open('files.devel.copy')
test_input_array = []
for line in f:
    line = line.split('\n')[0]
    input_file = '../features/str/' + line + '.str'
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
   y_test_pred = []
   for k, (i,o) in enumerate(dev_data):
        pred = vae.predict_label(i)
        prediction = np.argmax(pred.value())
        f.write(str(k) + ' ' + str(prediction) + ':' + str(ids[prediction]) +  ' ' + str(prediction) + ':' + str(ids[prediction])  + '\n')
   f.close()
     
 
def get_challenge_uar(epoch):
   cmd = 'perl format_pred.pl /home3/srallaba/data/ComParE2018_SelfAssessedAffect/arff/ComParE2018_SelfAssessedAffect.ComParE.devel.arff  submission_' + str(epoch) + '.txt submission.arff 6375'
   l.logger(cmd)
   os.system(cmd)

   cmd = 'perl score.pl /home3/srallaba/data/ComParE2018_SelfAssessedAffect/arff/ComParE2018_SelfAssessedAffect.ComParE.devel.arff submission.arff 6375'
   l.logger(cmd)
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


# Instantiate DNN and define the loss
m = dy.Model()
vae = SequenceVariationalAutoEncoder(m, input_dim, hidden, num_classes, 12, dy.selu) 
trainer = dy.AdamTrainer(m)
update_params = 32
num_epochs = 30

train_data = zip(x_train, y_train)
dev_data = zip(x_dev, y_dev)
num_train = len(train_data)
startTime = time.time()

# Loop over the training instances and call the mlp
for epoch in range(num_epochs):
  start_time = time.time()
  l.logger("Epoch = %d" %epoch)
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
       kl_loss, recons_loss  = vae.calc_loss_basic(i, o)
       loss = dy.esum([kl_loss, recons_loss])
       KL_loss += kl_loss.value()
       RECONS_loss += recons_loss.value()
       train_loss += loss.value()
       loss.backward()
       if frame_count % int(0.1*num_train) == 1:
           print "   Train Loss after processing " +  str(frame_count) + " number of files : " +  str(float(train_loss/frame_count))
       if frame_count % update_params == 1:
            trainer.update() 
  end_time = time.time()
  duration = end_time - start_time
  start_time = end_time
  l.logger("Train Loss after epoch " +  str(epoch) + " : " +  str(float(train_loss/frame_count)) + " with " + frame_count + " frames, in " +float((end_time - startTime)/60)  + " minutes "  )
  l.logger("KL Loss: " + str(float(KL_loss/frame_count)) + " RECONS loss: " + str(float(RECONS_loss/frame_count)))
  l.logger( "I think I will run for another " + str(float( duration * ( num_epochs - epoch) / 60 )) +  " minutes ")
  print '\n'

  test(epoch)
  get_challenge_uar(epoch)


