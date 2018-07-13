#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import CNN
from scipy import signal
import mtspec
from tqdm import tqdm



# Read from .mat file
# tr_eeg = loadmat('trimmedData.EEG.mat')
# y = pd.read_csv('label.csv', header=None).values
# EEG = tr_eeg['EEGtrimmed']
# EEG = np.moveaxis(EEG, -1, 0)
# np.save('EEG', EEG)

# # # convert integers to dummy variables (i.e. one hot encoded)
# encoder = LabelEncoder()
# encoder.fit(y)
# encoded_Y = encoder.transform(y)
# y = np_utils.to_categorical(encoded_Y)
# np.save('y', y)
#  0 | 1 | 2| 3| 4| 5| 6| 7 | 8 | 9 | 10|11|12|13|14|15| 16| 17| 18| 19| 20| 21 |22|23|24|25|26| 27|28|29|30| 31 |
# Fp1|Fp2|F7|F3|Fz|F4|F8|FC5|FC1|FC2|FC6|T7|C3|Cz|C4|T8|TP9|CP5|CP1|CP2|CP6|TP10|P7|P3|Pz|P4|P8|PO9|O1|Oz|O2|PO10|

# Read data
EEG = np.load('EEG.npy')
y = np.load('y.npy')


cwtmatr = np.abs(np.load('cwtmatr.npy'))
sft = np.abs(np.load('sft.npy'))
tf = np.load('tf.npy')


def reshape_1D_conv(X):
    X_rashaped = np.array([X[i, :, :].flatten()
                          for i in range(X.shape[0])])
    X_rashaped = X_rashaped.reshape(X_rashaped.shape[0],
                                    X_rashaped.shape[1],
                                    1)
    return X_rashaped


def reshape_2D_conv(X):
    X_reshaped = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    return X_reshaped
# # channels 0:22
# X1 = np.array([EEG[i, :, 0:22].flatten() for i in range(EEG.shape[0])])
# X1 = X1.reshape(X1.shape[0], X1.shape[1], 1)

# # channels 5:10
# X2 = np.array([EEG[i, :, 5:10].flatten() for i in range(EEG.shape[0])])
# X2 = X2.reshape(X1.shape[0], X2.shape[1], 1)

# # channels 7:121
# X3 = np.array([EEG[i, :, 7:21].flatten() for i in range(EEG.shape[0])])
# X3 = X3.reshape(X3.shape[0], X3.shape[1], 1)

# CNN.CNN1D(X1, y, epochs=500, name='ch0:22', no_GPU=4)
# CNN.CNN1D(X2, y, epochs=500, name='ch5:10', no_GPU=4)
# CNN.CNN1D(X3, y, epochs=500, name='ch7:21', no_GPU=4)


# ###########################################################
X_1D = reshape_1D_conv(cwtmatr)
CNN.CNN1D(X_1D, y, epochs=50, name='Wavelet_1D', no_GPU=4)
X_2D = reshape_2D_conv(cwtmatr)
CNN.CNN2D(X_2D, y, epochs=25, name='Wavelet_2D', no_GPU=4)

sft = np.abs(np.load('sft50.npy'))
X_1D = reshape_1D_conv(sft)
CNN.CNN1D(X_1D, y, epochs=50, name='sft50', no_GPU=4)
X_2D = reshape_2D_conv(sft)
CNN.CNN2D(X_2D, y, epochs=25, name='sft50', no_GPU=4)

sft = np.abs(np.load('sft100.npy'))
X_1D = reshape_1D_conv(sft)
CNN.CNN1D(X_1D, y, epochs=50, name='sft100', no_GPU=4)
X_2D = reshape_2D_conv(sft)
CNN.CNN2D(X_2D, y, epochs=25, name='sft100', no_GPU=4)

sft = np.abs(np.load('sft150.npy'))
X_1D = reshape_1D_conv(sft)
CNN.CNN1D(X_1D, y, epochs=50, name='sft150', no_GPU=4)
X_2D = reshape_2D_conv(sft)
CNN.CNN2D(X_2D, y, epochs=25, name='sft150', no_GPU=4)

tf = np.load('tf50.npy')
X_1D = reshape_1D_conv(tf)
CNN.CNN1D(X_1D, y, epochs=50, name='tf50', no_GPU=4)
X_2D = reshape_2D_conv(tf)
CNN.CNN2D(X_2D, y, epochs=25, name='tf50', no_GPU=4)

tf = np.load('tf100.npy')
X_1D = reshape_1D_conv(tf)
CNN.CNN1D(X_1D, y, epochs=50, name='tf100', no_GPU=4)
X_2D = reshape_2D_conv(tf)
CNN.CNN2D(X_2D, y, epochs=25, name='tf100', no_GPU=4)

tf = np.load('tf150.npy')
X_1D = reshape_1D_conv(tf)
CNN.CNN1D(X_1D, y, epochs=50, name='tf150', no_GPU=4)
X_2D = reshape_2D_conv(tf)
CNN.CNN2D(X_2D, y, epochs=25, name='tf150', no_GPU=4)