#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import CNN
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
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


num_GPU = 4
epochs = 100
verbose = 1


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
for i in range(12):
    X = np.load('./person/P'+str(i+1)+'X.npy')
    y = np.load('./person/P'+str(i+1)+'y.npy')
    X = reshape_1D_conv(X)
    CV.CNN1D(X, y, epochs=50, name='LSTM', optimizer='adam',
             batch_size=32, loss='categorical_crossentropy',
             metrics=['accuracy'], test_size=0.3, verbose=1)

    # LSTM
    X = np.abs(np.load('./person/P'+str(i+1)+'X.npy'))
    CNN.LSTMNN(X, y, epochs=50, name='LSTM', optimizer='adam',
               batch_size=32, loss='categorical_crossentropy',
               metrics=['accuracy'], test_size=0.3, verbose=1)

    # Short Time Fourier 1D
    print("Short Time Fourier 1D")
    X = np.abs(np.load('./person/P'+str(i+1)+'X.npy'))
    X = reshape_1D_conv(X)
    model = CNN.CNN1D(X, y, epochs=50, name='LSTM', optimizer='adam',
                      batch_size=32, loss='categorical_crossentropy',
                      metrics=['accuracy'], test_size=0.3, verbose=1)

    # Short Time Fourier 1D with log
    print("Short Time Fourier 1D with log")
    X = np.abs(np.load('stft-1D-50.npy'))
    X = np.log10(X.clip(min=000000.1))
    X = reshape_1D_conv(X)
    model = CNN.CNN1D(X, y, epochs=50, name='LSTM', optimizer='adam',
                      batch_size=32, loss='categorical_crossentropy',
                      metrics=['accuracy'], test_size=0.3, verbose=1)

    # Short Time Fourier 2D
    print("Short Time Fourier 2D")
    X = np.abs(np.load('stft-2D-50.npy'))
    model = CNN.CNN2D(X, y, epochs=50, name='LSTM', optimizer='adam',
                      batch_size=32, loss='categorical_crossentropy',
                      metrics=['accuracy'], test_size=0.3, verbose=1)

    # STFT 2D with Log Transformation
    X = np.abs(np.load('stft-2D-50.npy'))
    X = np.log10(X.clip(min=000000.1))
    model = CNN.CNN2D(X, y, epochs=50, name='LSTM', optimizer='adam',
                      batch_size=32, loss='categorical_crossentropy',
                      metrics=['accuracy'], test_size=0.3, verbose=1)

    print("Multitaper 1D")
    X = np.abs(np.load('mt-1D-50.npy'))
    X = reshape_1D_conv(X)
    model = CNN.CNN1D(X, y, epochs=50, name='LSTM', optimizer='adam',
                      batch_size=32, loss='categorical_crossentropy',
                      metrics=['accuracy'], test_size=0.3, verbose=1)

    # MultiTaper 1D with log
    X = np.abs(np.load('mt-1D-50.npy'))
    X = np.log10(X.clip(min=000000.1))
    X = reshape_1D_conv(X)
    model = CNN.CNN1D(X, y, epochs=50, name='LSTM', optimizer='adam',
                      batch_size=32, loss='categorical_crossentropy',
                      metrics=['accuracy'], test_size=0.3, verbose=1)

    # Multitaper 2D
    X = np.abs(np.load('mt-2D-50.npy'))
    model = CNN.CNN2D(X, y, epochs=50, name='LSTM', optimizer='adam',
                      batch_size=32, loss='categorical_crossentropy',
                      metrics=['accuracy'], test_size=0.3, verbose=1)

    # Multitaper 2D with Log
    X = np.abs(np.load('stft-2D-50.npy'))
    X = np.log10(X.clip(min=000000.1))
    model = CNN.CNN2D(X, y, epochs=50, name='LSTM', optimizer='adam',
                      batch_size=32, loss='categorical_crossentropy',
                      metrics=['accuracy'], test_size=0.3, verbose=1)
# ################ TRAIN MODELS #########################
# ConvLSTM2D
# X = np.load('EEG_2D_LSTM.npy')
# X = X.reshape(X.shape[0], X.shape[1], X.shape[2], X.shape[3], 1)
# CNN.ConvLSTM(X, y, epochs=epochs, name='ConvLSTM', num_GPU=num_GPU,
#            optimizer='adam', batch_size=64, loss='categorical_crossentropy',
#            metrics=['accuracy'], test_split_size=0.1, verbose=1)


# LSTM
# X = np.load('EEG.npy')
# print(X.shape)
# CNN.LSTMNN(X, y, epochs=10, name='LSTM', num_GPU=num_GPU,
#            optimizer='adam', batch_size=32, loss='categorical_crossentropy',
#            metrics=['accuracy'], test_split_size=0.3, verbose=1)
