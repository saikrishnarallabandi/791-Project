import os
import numpy as np

folder = '../../SoundNet-tensorflow/soundnet_feats_SAA_emphed/'
output = '../features/acoustic_embeddings'

files = sorted(os.listdir(folder))

for file in files:
    print file
    A = np.load(folder + '/' + file)
    a = A['arr_0']
    inp = np.mean(a[4],axis=0)
    g = open(output + '/' + file.split('.')[0] + '.txt', 'w')
    g.write(' '.join(str(k) for k in np.mean(a[4],axis=0)) + '\n')
    g.close()  
  
