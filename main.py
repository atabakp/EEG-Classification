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
np.save('EEG', EEG)

# convert integers to dummy variables (i.e. one hot encoded)
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
y = np_utils.to_categorical(encoded_Y)
np.save('y', y)
#  0 | 1 | 2| 3| 4| 5| 6| 7 | 8 | 9 | 10|11|12|13|14|15| 16| 17| 18| 19| 20| 21 |22|23|24|25|26| 27|28|29|30| 31 |
# Fp1|Fp2|F7|F3|Fz|F4|F8|FC5|FC1|FC2|FC6|T7|C3|Cz|C4|T8|TP9|CP5|CP1|CP2|CP6|TP10|P7|P3|Pz|P4|P8|PO9|O1|Oz|O2|PO10|

# Read data
EEG = np.load('EEG.npy')
y = np.load('y.npy')

# Wavelet spectogram
widths = np.arange(1, 40)
cwtmatr = np.stack([np.hstack([signal.cwt(EEG[j, :, i], signal.morlet, widths)
                    for i in range(EEG.shape[2])])
                    for j in range(EEG.shape[0])])

# Short-Time Fourier Transform
sft = np.stack([np.hstack([signal.stft(EEG[j, :, i], fs=100)[2]
               for i in range(EEG.shape[2])]) for j in range(EEG.shape[0])])

# Multitaper spectogram
tapers, _, _ = mtspec.dpss(npts=20, fw=3, number_of_tapers=5)
tf = np.stack([np.hstack(
    [np.mean(np.power(np.abs([signal.stft(EEG[j, :, i],
                              fs=100, window=tapers[:, t],
                              nperseg=tapers.shape[0])[2]
                              for t in range(tapers.shape[1])]), 2), axis=0)
        for i in range(EEG.shape[2])])
                              for j in range(EEG.shape[0])])


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
X_wavelet = cwtmatr.reshape(cwtmatr.shape[1], cwtmatr.shape[2], 1)
CNN.CNN2D(X_wavelet, y, epochs=500, name='wavelet_2D', no_GPU=4)

X_wave_1d = np.array([cwtmatr[i, :, :].flatten()
                     for i in range(cwtmatr.shape[0])])
X_wave_1d = X_wave_1d.reshape(X_wave_1d.shape[0], X1.shape[1], 1)
CNN.CNN1D(X3, y, epochs=500, name='wavelet_1D', no_GPU=4)
