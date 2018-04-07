from utils import *

import keras
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Dense, AlphaDropout
from keras.callbacks import *

import glog as log

def get_model(window = 5, num_classes = 3, input_dim = 50, hidden = 256):
    model = Sequential()
    model.add(LSTM(hidden, return_sequences=True, input_shape=(1601, input_dim)))
    model.add(LSTM(hidden, return_sequences=True))
    model.add(LSTM(hidden))
    model.add(Dense(hidden, activation='selu')) 
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    log.info('Loading labels')
    labels = load_labels('../data/ComParE2018_SelfAssessedAffect/lab/ComParE2018_SelfAssessedAffect.tsv')
    log.info('Loaded labels')

    log.info('Loading developing set')
    devel_input_array, devel_output_array = load_data('files.devel', labels)
    log.info('Loaded developing set')

    log.info('Loading training set')
    train_input_array, train_output_array = load_data('files.train', labels)
    log.info('Loaded training set')

    x_dev, y_dev = make_XY(devel_input_array, devel_output_array, input_dim, num_classes)
    x_train, y_train = make_XY(train_input_array, train_output_array, input_dim, num_classes)

    model = get_model()
    model.summary()
    model.fit(x_train, y_train, batch_size=64, epochs=6, shuffle=True, validation_data=(x_dev,y_dev))

# actuall content for modules
# from scripts.0_mvp_keras import get_model
# model = get_model()
# model.fit(x_train, y_train, batch_size=64, epochs=6, shuffle=True, validation_data=(x_dev,y_dev))

