import numpy as np

embeddings_file = 'embeddings'
labels_file = 'filenames'
fnames = []
f = open(labels_file)
for line in f:
    line = line.split('\n')[0]
    fnames.append(line)

folder = '../features/text_embeddings'


# Add text embeddings
A = np.loadtxt(embeddings_file)
for i,a in enumerate(A):
   fname = fnames[i]
   g = open(folder +  '/' + fname + '.txt','w')
   g.write(' '.join(str(k) for k in a) + '\n')
   g.close()
