#from utils import *
from torch.utils.data import Dataset
import torch.utils.data as data_utils
from collections import defaultdict
import numpy as np
import random
from keras.utils import to_categorical
from torch.utils.data import Dataset
import torch.utils.data as data_utils
from collections import defaultdict
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from keras.utils import to_categorical
import torch

### Help:  Piazza posts: https://piazza.com/class/j9xdaodf6p1443?cid=2774

window = 5
num_classes = 3
input_dim = 50
hidden = 256
batch_size = 2

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
    input_file = '../data/mcep_ascii/' + line + '.mcep'
    inp = np.loadtxt(input_file)
    devel_input_array.append(inp)
    devel_output_array.append(labels[line])



class COMPARE(Dataset):

    def __init__(self, A,B):
      
        self.input = A
        self.output = B

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return self.input[idx], self.output[idx]


from torch.utils.data import Dataset
import torch.utils.data as data_utils
from collections import defaultdict

dataset = COMPARE(devel_input_array,devel_output_array)
train_loader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True)



def CreateOnehotVariable( input_x, encoding_dim=63):
    if type(input_x) is Variable:
        input_x = input_x.data 
    input_type = type(input_x)
    batch_size = input_x.size(0)
    time_steps = input_x.size(1)
    input_x = input_x.unsqueeze(2).type(torch.LongTensor)
    onehot_x = Variable(torch.LongTensor(batch_size, time_steps, encoding_dim).zero_().scatter_(-1,input_x,1)).type(input_type)
    
    return onehot_x

class decoder(nn.Module):
   
    def __init__(self, embedding_dim, hidden_dim, target_size, max_label_length=100):
       super(decoder, self).__init__()
       self.hidden_dim = hidden_dim
       self.embedding_dim = embedding_dim
       self.target_size = target_size
       self.float_type = torch.torch.cuda.FloatTensor
       self.max_label_length = max_label_length
       self.rnn_layer = nn.LSTM(embedding_dim+hidden_dim*2,hidden_dim*2,num_layers=1)
       self.attention = Attention( mlp_preprocess_input=True, preprocess_mlp_dim=128,
                                    activate='relu', input_feature_dim=hidden_dim)
       self.softmax = nn.LogSoftmax(dim=-1)
       self.character_distribution = nn.Linear(hidden_dim*4,embedding_dim)

    def fwd_step(self,input_word, last_hidden_state,listener_feature):
        rnn_output, hidden_state = self.rnn_layer(input_word,last_hidden_state)
        attention_score, context = self.attention(rnn_output,listener_feature)
        #print("Returned from Attention")
        concat_feature = torch.cat([rnn_output.squeeze(dim=1),context],dim=-1)
        #print("Shape of concat feature", concat_feature.shape)
        raw_pred = self.softmax(self.character_distribution(concat_feature))
        return raw_pred, hidden_state, context, attention_score
    
    def forward(self, context_tensor, ground_truth):
      
       output_word = CreateOnehotVariable(self.float_type(np.zeros((context_tensor.shape[0],1))),embedding_dim)
       rnn_input = torch.cat([output_word.cuda(),context_tensor[:,0:1,:]],dim=-1)

       if print_flag == 1:
         print("  Shape of context matrix is ", context_tensor.shape)
         print("  Shape of word is ", output_word.shape)
         print("  Shape of RNN Input is ", rnn_input.shape)

       max_len = ground_truth.shape[1]
       hidden_state = None
       attention_tensor = []
       prediction_tensor = []
       for i in range(max_len):
           probs,hidden_state,context,attention = self.fwd_step(rnn_input, hidden_state, context_tensor)
           if print_flag == 1:
             print("Currently in timestep", i+1)
             print("  Shape of attention tensor: ", attention.shape)
             print("  Shape of probs is ", probs.shape)
             print("  Shape of context: ", context.shape)
           attention_tensor.append(attention)
           prediction = probs.unsqueeze(1)
           prediction_tensor.append(probs)
           if print_flag  == 1:
             print("  Shape of prediction: ", prediction.shape)
             print("  Shape of context tensor: ", context_tensor.shape)
 
           if teacher_forcing == 1:
               rnn_input = torch.cat([ground_truth[:,i:i+1,:].type(self.float_type), context_tensor[:,0:1,:]],dim=-1)
           else:
               rnn_input = torch.cat([prediction, context_tensor[:,0:1,:]],dim=-1)
           if  print_flag  == 1:
             print("Time step done", i+1)
             print('\n')

       return prediction_tensor


class Attention(nn.Module):
    def __init__(self, mlp_preprocess_input, preprocess_mlp_dim, activate, input_feature_dim=512):
        super(Attention,self).__init__()
        self.mlp_preprocess_input = False # mlp_preprocess_input
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        if mlp_preprocess_input:
            self.preprocess_mlp_dim  = preprocess_mlp_dim
            self.phi = nn.Linear(input_feature_dim,preprocess_mlp_dim)
            self.psi = nn.Linear(input_feature_dim,preprocess_mlp_dim)
            self.activate = getattr(F,activate)

    def forward(self, decoder_state, listener_feature):
        #print("In attention")
        if self.mlp_preprocess_input:

            energy = torch.bmm(comp_decoder_state,comp_listener_feature).squeeze(dim=1)
        attention_score = self.softmax(energy)
        context = torch.sum(listener_feature*attention_score.unsqueeze(2).repeat(1,1,listener_feature.size(2)),dim=1)

        return attention_score,context



class encoder(nn.Module):
     
    def __init__(self, input_dim, hidden_dim):
       super(encoder, self).__init__()
       self.hidden_dim = hidden_dim
       self.lstm = nn.LSTM(input_dim, hidden_dim, 1,  batch_first = True, bidirectional=True)

    def forward(self, utterance_batch):
        batch_size = utterance_batch.shape[0]
        time_steps = utterance_batch.shape[1]
        feature_dim = utterance_batch.shape[2]
        x_input = utterance_batch.contiguous()
        lstm_out, self.hidden = self.lstm(x_input, None)
        return lstm_out

embedding_dim = 3
hidden_dim = 128
vocab_size = 3
target_size = vocab_size
input_dim = 39
print("The target size is ", target_size)
baseline_encoder = encoder(input_dim, hidden_dim)
baseline_decoder = decoder(embedding_dim, hidden_dim, vocab_size)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(list(baseline_encoder.parameters()) + list( baseline_decoder.parameters()), lr=0.01)
objective = nn.CrossEntropyLoss(ignore_index=0)

def train():
 total_loss = 0
 for ctr, t in enumerate(train_loader):
    a,b = t
    #print(a.shape, b.shape)
    a,b = Variable(torch.from_numpy(a)), Variable(torch.from_numpy(b), requires_grad=False)
    #a,b = a.cuda(), b.cuda()
    pred = baseline_listener(a)
    #print("Shape of listener output:", pred.shape)
    #print("Batch Done")
    words = baseline_speller(pred, b)
    pred_y = torch.cat([torch.unsqueeze(each_y,1) for each_y in words],1).view(-1,embedding_dim)
    true_y = torch.max(b,dim=2)[1].view(-1)
    true_y = true_y.detach()
    #print("Shape of words:", pred_y.shape)
    #print("Shape of Truth: ", true_y.shape)
    loss = objective(pred_y, true_y)
    #print('\n')
    #sys.exit()
    total_loss += loss.cpu().data.numpy()
    if ctr % 4 == 1 :
       print(" Loss after ", ctr+1, "utterances: ", total_loss/ (ctr+1) * 1.0 * batch_size)

    loss.backward()
    optimizer.step()
    


for epoch in range(10):
    print("Running epoch ", epoch)
    train()
