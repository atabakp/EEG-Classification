#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import CNN

num_GPU = 0
epochs = 2
verbose = 1
#  0 | 1 | 2| 3| 4| 5| 6| 7 | 8 | 9 | 10|11|12|13|14|15| 16| 17| 18| 19| 20| 21 |22|23|24|25|26| 27|28|29|30| 31 |
# Fp1|Fp2|F7|F3|Fz|F4|F8|FC5|FC1|FC2|FC6|T7|C3|Cz|C4|T8|TP9|CP5|CP1|CP2|CP6|TP10|P7|P3|Pz|P4|P8|PO9|O1|Oz|O2|PO10|


def reshape_1D_conv(X):
    X_rashaped = np.array([X[i, :, :].flatten()
                           for i in range(X.shape[0])])  # concatenate
    X_rashaped = X_rashaped.reshape(X_rashaped.shape[0],
                                    X_rashaped.shape[1],
                                    1)  # add 3rd dimension
    return X_rashaped


def reshape_2D_conv(X):
    X_reshaped = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    return X_reshaped


# Read data
y = np.load('y.npy')
# ################ TRAIN MODELS #########################

# channels all
X = np.load('EEG.npy')
X = reshape_1D_conv(X)
CNN.CNN1D(X, y, epochs=epochs, name='TimeDomain-allch', num_GPU=num_GPU,
          optimizer='adam', batch_size=32, loss='binary_crossentropy',
          metrics=['accuracy'], test_split_size=0.1 ,verbose=verbose)


# channels 7:21
X = np.load('EEG.npy')
X = reshape_1D_conv(X[:, :, 7:21])
CNN.CNN1D(X, y, epochs=epochs, name='TimeDomain-ch0:22', num_GPU=num_GPU,
          optimizer='adam', batch_size=32, loss='binary_crossentropy',
          metrics=['accuracy'], test_split_size=0.1 ,verbose=verbose)

# Short Time Fourier 1D
X = np.abs(np.load('stft-1D-50.npy'))
X = reshape_1D_conv(X)
model = CNN.CNN1D(X, y, epochs=epochs, name='stft-1D-50', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1 ,verbose=verbose)

X = np.abs(np.load('stft-1D-100.npy'))
X = reshape_1D_conv(X)
model = CNN.CNN1D(X, y, epochs=epochs, name='stft-1D-100', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1 ,verbose=verbose)

X = np.abs(np.load('stft-1D-150.npy'))
X = reshape_1D_conv(X)
model = CNN.CNN1D(X, y, epochs=epochs, name='stft-1D-150', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1 ,verbose=verbose)

# Short Time Fourier 1D with log
X = np.log10(np.abs(np.load('stft-1D-50.npy')))
X = reshape_1D_conv(X)
model = CNN.CNN1D(X, y, epochs=epochs, name='stft-1D-50-log', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1 ,verbose=verbose)

X = np.log10(np.abs(np.load('stft-1D-100.npy')))
X = reshape_1D_conv(X)
model = CNN.CNN1D(X, y, epochs=epochs, name='stft-1D-100-log', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1 ,verbose=verbose)

X = np.log10(np.abs(np.load('stft-1D-150.npy')))
X = reshape_1D_conv(X)
model = CNN.CNN1D(X, y, epochs=epochs, name='stft-1D-150-log', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1 ,verbose=verbose)


# Short Time Fourier 2D
X = np.abs(np.load('stft-2D-50.npy'))
model = CNN.CNN2D(X, y, epochs=epochs, name='stft-2D-50', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1 ,verbose=verbose)

X = np.abs(np.load('stft-2D-100.npy'))
model = CNN.CNN2D(X, y, epochs=epochs, name='stft-2D-100', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1 ,verbose=verbose)

X = np.abs(np.load('stft-2D-100.npy'))
model = CNN.CNN2D(X, y, epochs=epochs, name='stft-2D-150', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1 ,verbose=verbose)

# STFT 2D with Log Transformation
X = np.log10(np.abs(np.load('stft-2D-50.npy')))
model = CNN.CNN2D(X, y, epochs=epochs, name='stft-2D-50-log', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1 ,verbose=verbose)

X = np.log10(np.abs(np.load('stft-2D-100.npy')))
model = CNN.CNN2D(X, y, epochs=epochs, name='stft-2D-100-log', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1 ,verbose=verbose)

X = np.log10(np.abs(np.load('stft-2D-150.npy')))
model = CNN.CNN2D(X, y, epochs=epochs, name='stft-2D-150-log', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1 ,verbose=verbose)


# Multitaper 1D
X = np.abs(np.load('mt-1D-50'))
X = reshape_1D_conv(X)
model = CNN.CNN1D(X, y, epochs=epochs, name='mt-1D-50', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1 ,verbose=verbose)

X = np.abs(np.load('mt-1D-100.npy'))
X = reshape_1D_conv(X)
model = CNN.CNN1D(X, y, epochs=epochs, name='mt-1D-100', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1 ,verbose=verbose)

X = np.abs(np.load('mt-1D-150.npy'))
X = reshape_1D_conv(X)
model = CNN.CNN1D(X, y, epochs=epochs, name='mt-1D-150', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1 ,verbose=verbose)

# MultiTaper 1D with log
X = np.log10(np.abs(np.load('mt-1D-50.npy')))
X = reshape_1D_conv(X)
model = CNN.CNN1D(X, y, epochs=epochs, name='mt-1D-50-log', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1 ,verbose=verbose)

X = np.log10(np.abs(np.load('mt-1D-100.npy')))
X = reshape_1D_conv(X)
model = CNN.CNN1D(X, y, epochs=epochs, name='mt-1D-100-log', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1 ,verbose=verbose)

X = np.log10(np.abs(np.load('mt-1D-150.npy')))
X = reshape_1D_conv(X)
model = CNN.CNN1D(X, y, epochs=epochs, name='mt-1D-150-log', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1 ,verbose=verbose)


# Multitaper 2D
X = np.abs(np.load('mt-2D-50.npy'))
model = CNN.CNN2D(X, y, epochs=epochs, name='mt-2D-50', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1 ,verbose=verbose)

X = np.abs(np.load('mt-2D-100.npy'))
model = CNN.CNN2D(X, y, epochs=epochs, name='mt-1D-100', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1 ,verbose=verbose)

X = np.abs(np.load('mt-2D-150.npy'))
model = CNN.CNN2D(X, y, epochs=epochs, name='mt-1D-150', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1 ,verbose=verbose)

# Multitaper 2D with Log
X = np.log10(np.abs(np.load('stft-2D-50.npy')))
model = CNN.CNN2D(X, y, epochs=epochs, name='stft-2D-50-log', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1 ,verbose=verbose)

X = np.log10(np.abs(np.load('stft-2D-100.npy')))
model = CNN.CNN2D(X, y, epochs=epochs, name='stft-2D-100-log', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1 ,verbose=verbose)

X = np.log10(np.abs(np.load('stft-2D-150.npy')))
model = CNN.CNN2D(X, y, epochs=epochs, name='stft-2D-150-log', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1 ,verbose=verbose)


# wavelet
X = np.log10(np.abs(np.load('cwt-1D-5.npy')))
X = reshape_1D_conv(X)
model = CNN.CNN1D(X, y, epochs=epochs, name='cwt-1D-5-log', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1 ,verbose=verbose)

X = np.abs(np.load('cwt-2D-5.npy'))
model = CNN.CNN2D(X, y, epochs=epochs, name='cwt-2D-5', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1 ,verbose=verbose)
