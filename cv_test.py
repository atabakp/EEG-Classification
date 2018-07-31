#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import CV
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import (Dense, Dropout, Conv1D, GlobalAveragePooling1D,
                          MaxPooling1D, Flatten, Conv2D, Activation,
                          MaxPooling2D, BatchNormalization, LSTM)
from keras import callbacks
import datetime
import os
from keras.models import Model
from keras import layers
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

num_GPU = 4
epochs = 100
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
# ConvLSTM2D
# X = np.load('EEG_2D_LSTM.npy')
# X = X.reshape(X.shape[0], X.shape[1], X.shape[2], X.shape[3], 1)
# CNN.ConvLSTM(X, y, epochs=epochs, name='ConvLSTM', num_GPU=num_GPU,
#            optimizer='adam', batch_size=64, loss='categorical_crossentropy',
#            metrics=['accuracy'], test_split_size=0.1, verbose=1)


# LSTM
# X = np.load('EEG.npy')
# X = reshape_1D_conv(X)
# CNN.LSTMNN(X, y, epochs=5, name='LSTM', num_GPU=num_GPU,
#            optimizer='adam', batch_size=32, loss='categorical_crossentropy',
#            metrics=['accuracy'], test_split_size=0.1, verbose=1)
# Print("Done!!!!!!!!!!!!!!!!!")

# channels all
X = np.load('EEG.npy')
X = reshape_1D_conv(X)
estimator = KerasClassifier(build_fn=CV.CNN1D, epochs=150, batch_size=10,
                            verbose=0)

kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print("Time all channels")
print("Mean (STD): %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# channels 7:21
X = np.load('EEG.npy')
X = reshape_1D_conv(X[:, :, 7:21])
estimator = KerasClassifier(build_fn=CV.CNN1D, epochs=150, batch_size=10,
                            verbose=0)

kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print("Time channels 7:21")
print("Mean (STD): %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# Short Time Fourier 1D
X = np.abs(np.load('stft-1D-50.npy'))
X = reshape_1D_conv(X)
estimator = KerasClassifier(build_fn=CV.CNN1D, epochs=150, batch_size=10,
                            verbose=0)

kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print("STFT-50 1D")
print("Mean (STD): %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

X = np.abs(np.load('stft-1D-100.npy'))
X = reshape_1D_conv(X)
estimator = KerasClassifier(build_fn=CV.CNN1D, epochs=150, batch_size=10,
                            verbose=0)

kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print("STFT-100 1D")
print("Mean (STD): %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

X = np.abs(np.load('stft-1D-150.npy'))
X = reshape_1D_conv(X)
estimator = KerasClassifier(build_fn=CV.CNN1D, epochs=150, batch_size=10,
                            verbose=0)

kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print("STFT-150 1D")
print("Mean (STD): %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# Short Time Fourier 1D with log
print("Short Time Fourier 1D with log")
X = np.abs(np.load('stft-1D-50.npy'))
X = np.log10(X.clip(min=000000.1))
X = reshape_1D_conv(X)
estimator = KerasClassifier(build_fn=CV.CNN1D, epochs=150, batch_size=10,
                            verbose=0)

kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print("STFT-50 log 1D")
print("Mean (STD): %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

X = np.abs(np.load('stft-1D-100.npy'))
X = np.log10(X.clip(min=000000.1))
X = reshape_1D_conv(X)
estimator = KerasClassifier(build_fn=CV.CNN1D, epochs=150, batch_size=10,
                            verbose=0)

kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print("STFT-100 log 1D")
print("Mean (STD): %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

X = np.abs(np.load('stft-1D-150.npy'))
X = np.log10(X.clip(min=000000.1))
X = reshape_1D_conv(X)
estimator = KerasClassifier(build_fn=CV.CNN1D, epochs=150, batch_size=10,
                            verbose=0)

kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print("STFT-150 log 1D")
print("Mean (STD): %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# Short Time Fourier 2D
print("Short Time Fourier 2D")
X = np.abs(np.load('stft-2D-50.npy'))
estimator = KerasClassifier(build_fn=CV.CNN2D, epochs=150, batch_size=10,
                            verbose=0)

kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print("STFT-50 log 2D")
print("Mean (STD): %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

X = np.abs(np.load('stft-2D-100.npy'))
estimator = KerasClassifier(build_fn=CV.CNN2D, epochs=150, batch_size=10,
                            verbose=0)

kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print("STFT-100 log 2D")
print("Mean (STD): %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

X = np.abs(np.load('stft-2D-100.npy'))
estimator = KerasClassifier(build_fn=CV.CNN2D, epochs=150, batch_size=10,
                            verbose=0)

kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print("STFT-150 log 2D")
print("Mean (STD): %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# STFT 2D with Log Transformation
X = np.abs(np.load('stft-2D-50.npy'))
X = np.log10(X.clip(min=000000.1))
estimator = KerasClassifier(build_fn=CV.CNN2D, epochs=150, batch_size=10,
                            verbose=0)

kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print("STFT-50 log 2D")
print("Mean (STD): %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

X = np.abs(np.load('stft-2D-100.npy'))
X = np.log10(X.clip(min=000000.1))
estimator = KerasClassifier(build_fn=CV.CNN2D, epochs=150, batch_size=10,
                            verbose=0)

kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print("STFT-100 log 2D")
print("Mean (STD): %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

X = np.abs(np.load('stft-2D-150.npy'))
X = np.log10(X.clip(min=000000.1))
estimator = KerasClassifier(build_fn=CV.CNN2D, epochs=150, batch_size=10,
                            verbose=0)

kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print("STFT-150 log 2D")
print("Mean (STD): %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# Multitaper 1D
print("Multitaper 1D")
X = np.abs(np.load('mt-1D-50.npy'))
X = reshape_1D_conv(X)
estimator = KerasClassifier(build_fn=CV.CNN1D, epochs=150, batch_size=10,
                            verbose=0)

kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print("MT 50 1D")
print("Mean (STD): %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

X = np.abs(np.load('mt-1D-100.npy'))
X = reshape_1D_conv(X)
estimator = KerasClassifier(build_fn=CV.CNN1D, epochs=150, batch_size=10,
                            verbose=0)

kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print("MT 100 1D")
print("Mean (STD): %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

X = np.abs(np.load('mt-1D-150.npy'))
X = reshape_1D_conv(X)
estimator = KerasClassifier(build_fn=CV.CNN1D, epochs=150, batch_size=10,
                            verbose=0)

kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print("MT 150 1D")
print("Mean (STD): %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# MultiTaper 1D with log
X = np.abs(np.load('mt-1D-50.npy'))
X = np.log10(X.clip(min=000000.1))
X = reshape_1D_conv(X)
estimator = KerasClassifier(build_fn=CV.CNN1D, epochs=150, batch_size=10,
                            verbose=0)

kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print("MT 50 log 1D")
print("Mean (STD): %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

X = np.abs(np.load('mt-1D-100.npy'))
X = np.log10(X.clip(min=000000.1))
X = reshape_1D_conv(X)
estimator = KerasClassifier(build_fn=CV.CNN1D, epochs=150, batch_size=10,
                            verbose=0)

kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print("MT 100 log 1D")
print("Mean (STD): %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

X = np.abs(np.load('mt-1D-150.npy'))
X = np.log10(X.clip(min=000000.1))
X = reshape_1D_conv(X)
estimator = KerasClassifier(build_fn=CV.CNN1D, epochs=150, batch_size=10,
                            verbose=0)

kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print("MT 150 log 1D")
print("Mean (STD): %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# Multitaper 2D
X = np.abs(np.load('mt-2D-50.npy'))
estimator = KerasClassifier(build_fn=CV.CNN2D, epochs=150, batch_size=10,
                            verbose=0)

kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print("MT 50 2D")
print("Mean (STD): %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

X = np.abs(np.load('mt-2D-100.npy'))
estimator = KerasClassifier(build_fn=CV.CNN2D, epochs=150, batch_size=10,
                            verbose=0)

kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print("MT 100 2D")
print("Mean (STD): %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

X = np.abs(np.load('mt-2D-150.npy'))
estimator = KerasClassifier(build_fn=CV.CNN2D, epochs=150, batch_size=10,
                            verbose=0)

kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print("MT 150 2D")
print("Mean (STD): %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# Multitaper 2D with Log
X = np.abs(np.load('stft-2D-50.npy'))
X = np.log10(X.clip(min=000000.1))
estimator = KerasClassifier(build_fn=CV.CNN2D, epochs=150, batch_size=10,
                            verbose=0)

kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print("MT 50 log 2D")
print("Mean (STD): %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

X = np.abs(np.load('stft-2D-100.npy'))
X = np.log10(X.clip(min=000000.1))
estimator = KerasClassifier(build_fn=CV.CNN2D, epochs=150, batch_size=10,
                            verbose=0)

kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print("MT 100 log 2D")
print("Mean (STD): %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

X = np.abs(np.load('stft-2D-150.npy'))
X = np.log10(X.clip(min=000000.1))
estimator = KerasClassifier(build_fn=CV.CNN2D, epochs=150, batch_size=10,
                            verbose=0)

kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print("MT 150 log 2D")
print("Mean (STD): %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


