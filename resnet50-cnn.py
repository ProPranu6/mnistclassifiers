import numpy as np
import warnings
from PIL import Image, ImageOps
from mlxtend.data import loadlocal_mnist
import math
import h5py
import scipy
from scipy import ndimage
import tensorflow as tf

from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Add
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils import to_categorical
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from mlxtend.data import loadlocal_mnist
from keras.initializers import glorot_uniform

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

%matplotlib inline


pathXtrain = '/content/drive/My Drive/train-images.idx3-ubyte'
pathYtrain = '/content/drive/My Drive/train-labels.idx1-ubyte'

pathXtest = '/content/drive/My Drive/t10k-images.idx3-ubyte'
pathYtest = '/content/drive/My Drive/t10k-labels.idx1-ubyte'

Xtrain_orig, Ytrain_orig = loadlocal_mnist(pathXtrain,pathYtrain) 
Xtest_orig, Ytest_orig = loadlocal_mnist(pathXtest,pathYtest)

Xtrain_orig = Xtrain_orig[:3000]
Ytrain_orig = Ytrain_orig[:3000]


Xtrain_orig = Xtrain_orig.reshape(-1,28,28)
Xtest_orig = Xtest_orig.reshape(-1,28,28)

Xtrain = Xtrain_orig.reshape(-1,28,28,1)/255
Xtest = Xtest_orig.reshape(-1,28,28,1)/255
Ytrain = to_categorical(Ytrain_orig, 10)
Ytest = to_categorical(Ytest_orig, 10)

print((Xtrain.shape))
print((Ytrain.shape))
imshow(Xtrain_orig[6], cmap = 'gray')

#Architecture : [Input]-[conv-relu-pool]-[convBlock]-([identityBlock]x3)-....

def identityBlock(aPrev, kernelSize, filters):
  aSave = aPrev

  kSz1, kSz2, kSz3 = kernelSize
  F1, F2, F3 = filters
  a = Conv2D(F1, kSz1, strides=(1,1), padding='same')(aPrev)
  a = BatchNormalization(axis = 3)(a)
  a = Activation(activation='relu')(a)

  a = Conv2D(F2, kSz2, strides=(1,1), padding='same')(a)
  a = BatchNormalization(axis = 3)(a)
  a = Activation(activation='relu')(a)

  a = Conv2D(F3, kSz3, strides=(1,1), padding='same')(a)
  a = BatchNormalization(axis = 3)(a)

  a = Add()([a, aSave])
  a = Activation(activation='relu')(a)

  return a

def convBlock(aPrev, kernelSize, filters, shortCutStride):
  aSave = aPrev

  kSz1, kSz2, kSz3 = kernelSize
  F1, F2, F3 = filters

  a = Conv2D(F1, (1,1), strides=(shortCutStride,shortCutStride))(aPrev)
  a = BatchNormalization(axis = 3)(a)
  a = Activation(activation='relu')(a)

  a = Conv2D(F2, kSz2, strides=(1,1), padding='same')(a)
  a = BatchNormalization(axis = 3)(a)
  a = Activation(activation='relu')(a)

  a = Conv2D(F3, kSz3, strides=(1,1), padding='same')(a)
  a = BatchNormalization(axis = 3)(a)

  aSave = Conv2D(F3, (1,1), strides= (shortCutStride, shortCutStride))(aSave)
  aSave = BatchNormalization(axis=3)(aSave)

  a = Add()([a, aSave])
  a = Activation(activation='relu')(a)

  return a


def resNet50(xtrain, ytrain, xtest, ytest, optimiser = 'adam', costing = 'categorical_crossentropy', evalMetrics=['accuracy'], epochs = 40, batchSize = 1000):
    X_Input = Input(xtrain.shape[1:])
    X = ZeroPadding2D(padding = (3,3))(X_Input)

    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convBlock(X, [3,3,3], filters=[64, 64, 256], shortCutStride=1)
    X = identityBlock(X, [3,3,3], [64, 64, 256])
    X = identityBlock(X, [3,3,3], [64, 64, 256])

    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    X = convBlock(X, [3,3,3], filters=[128, 128, 512], shortCutStride=2)
    X = identityBlock(X, [3,3,3], [128, 128, 512])
    X = identityBlock(X, [3,3,3], [128, 128, 512])
    X = identityBlock(X, [3,3,3], [128, 128, 512])

    # Stage 4 (≈6 lines)
    X = convBlock(X, [3,3,3], filters=[256, 256, 1024], shortCutStride=2)
    X = identityBlock(X, [3,3,3], [256, 256, 1024])
    X = identityBlock(X, [3,3,3], [256, 256, 1024])
    X = identityBlock(X, [3,3,3], [256, 256, 1024])
    X = identityBlock(X, [3,3,3], [256, 256, 1024])
    X = identityBlock(X, [3,3,3], [256, 256, 1024])

    # Stage 5 (≈3 lines)
    X = X = convBlock(X, [3,3,3], filters=[512, 512, 2048],shortCutStride=2)
    X = identityBlock(X, [3,3,3], [512, 512, 2048])
    X = identityBlock(X, [3,3,3], [512, 512, 2048])

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)

    X = Flatten()(X)
    X = Dense(10, activation='softmax', kernel_initializer=glorot_uniform(seed=0))(X)

    resNet50M = Model(X_Input,X)
    resNet50M.compile(optimiser, costing, metrics = evalMetrics)

    resNet50M.fit(xtrain, ytrain, epochs = epochs, batch_size = batchSize)

    preds = resNet50M.evaluate(xtest,ytest, batch_size = batchSize, verbose=1, sample_weight=None)

    print()
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))
  
    return "Done"



resNet50(Xtrain, Ytrain, Xtest, Ytest, epochs=5)
