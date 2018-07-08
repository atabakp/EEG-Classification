#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import CNN

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
X = np.array([EEG[i, :, :].flatten() for i in range(EEG.shape[0])])
X = X.reshape(X.shape[0], X.shape[1], 1)

CNN.CNN1D(X, y, epochs=20, name='1')
