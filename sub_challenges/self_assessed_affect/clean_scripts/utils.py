import numpy as np
from keras.utils import to_categorical

def get_max_len(arr):
   '''
   This takes a list of lists as input and returns the maximum length
   '''
   return max([len(a) for a in arr])


def load_labels(labels_file):
    labels = {}
    ids = ['l','m','h']
    cnt = 0 
    with open(labels_file) as f:
        for line in f:
            if cnt == 0:
                cnt+= 1
            else:
                line = line.split('\n')[0].split()
                fname = line[0].split('.')[0]
                lbl = ids.index(line[1])
                labels[fname] = lbl
    return labels


def load_data(file_name, labels):
    with open(file_name) as f:
        input_array = []
        output_array = []
        for line in f:
            line = line.split('\n')[0]
            input_file = '../data/mcep_ascii/' + line + '.mcep'
            inp = np.loadtxt(input_file)
            input_array.append(inp)
            output_array.append(labels[line])
        return input_array, output_array


def make_XY(input_array, output_array, input_dim, num_classes):
    x_dev = np.zeros( (len(input_array), 1601, input_dim), 'float32')
    y_dev = np.zeros( (len(input_array), num_classes ), 'float32')

    for i, (x,y) in enumerate(zip(input_array, output_array)):
       x_dev[i] = x
       y_dev[i] = to_categorical(y,num_classes)   
