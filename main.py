#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import CNN

# Read from .mat file
tr_eeg = loadmat('trimmedData.EEG.mat')
y = pd.read_csv('label.csv', header=None).values
EEG = tr_eeg['EEGtrimmed']
EEG = np.moveaxis(EEG, -1, 0)


# convert integers to dummy variables (i.e. one hot encoded)
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
y = np_utils.to_categorical(encoded_Y)

# concatinate all 32 channels
X1 = np.array([EEG[i, :, :].flatten() for i in range(EEG.shape[0])])
X1 = X1.reshape(X.shape[0], X1.shape[1], 1)

# channels 5:10
X2 = np.array([EEG[i, :, 5:10].flatten() for i in range(EEG.shape[0])])
X2 = X2.reshape(X.shape[0], X2.shape[1], 1)

CNN.CNN1D(X1, y, epochs=500, name='allch')
CNN.CNN1D(X2, y, epochs=500, name='ch5:10')
