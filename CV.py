#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from keras.models import Sequential
from keras.layers import (Dense, Dropout, Conv1D, GlobalAveragePooling1D,
                          MaxPooling1D, Flatten, Conv2D, Activation,
                          MaxPooling2D, BatchNormalization, LSTM, ConvLSTM2D)
from keras import callbacks
import datetime
import os
from keras.models import Model
from keras import layers
import keras
import csv
from sklearn.model_selection import StratifiedKFold
from keras.utils import to_categorical
from keras.utils import multi_gpu_model


def CNN1D(X, y, epochs, name, folds=10, verbose=1, batch_size=32,
          optimizer='adam', loss='categorical_crossentropy',
          metrics=['accuracy']):

    y = np.argmax(y, axis=1)
    kfold = StratifiedKFold(n_splits=folds, shuffle=True)
    cvscores = []
    i = 1
    for train, val in kfold.split(X, y):
        model = Sequential()
        model.add(Conv1D(filters=1, kernel_size=5, strides=10,
                         input_shape=(X.shape[1], 1),
                         kernel_initializer='uniform',
                         name='1-Conv1D'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5, name='2-dropout'))
        model.add(MaxPooling1D(2, name='3-maxpooling'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(3, activation='softmax'))

        y_tr = to_categorical(y[train])

        parallel_model = multi_gpu_model(model, gpus=4)
        parallel_model.compile(loss=loss, optimizer=optimizer,
                               metrics=metrics)
        parallel_model.fit(X[train], y_tr, epochs=epochs,
                           batch_size=batch_size, verbose=verbose)
        y_val = to_categorical(y[val])
        scores = parallel_model.evaluate(X[val], y_val, verbose=verbose)
        print("fold %d: %s: %.2f%%" % (i, parallel_model.metrics_names[1],
                                       scores[1]*100))
        i = i+1
        cvscores.append(scores[1] * 100)
    np.savetxt(name + ".csv", cvscores, delimiter=",")
    print(name + " %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


def CNN2D(X, y, epochs, name, folds=10, verbose=1, batch_size=32,
          optimizer='adam', loss='categorical_crossentropy',
          metrics=['accuracy']):

    y = np.argmax(y, axis=1)
    kfold = StratifiedKFold(n_splits=folds, shuffle=True)
    cvscores = []
    i = 1
    for train, val in kfold.split(X, y):
        model = Sequential()
        model.add(Conv2D(64, (2, 1), data_format="channels_first",
                  input_shape=(X.shape[1], X.shape[2], X.shape[3]),
                  strides=[2, 1]))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2D(32, (2, 1), data_format="channels_first",
                  strides=[2, 1]))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2D(16, (2, 1), data_format="channels_first",
                  strides=[2, 1]))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # model.add(Conv2D(5, (3, 2), data_format="channels_first",
        #           strides=[2, 1]))
        # model.add(BatchNormalization())
        # model.add(Activation('relu'))

        # model.add(Conv2D(5, (3, 2), data_format="channels_first",
        #           strides=[2, 1]))
        # model.add(BatchNormalization())
        # model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        # model.add(Dense(512, activation='relu'))
        # model.add(Dense(512, activation='relu'))
        # model.add(Dense(512, activation='relu'))
        model.add(Dense(2000, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(800, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(500, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(3, activation='softmax'))

        y_tr = to_categorical(y[train])

        parallel_model = multi_gpu_model(model, gpus=4)
        parallel_model.compile(loss=loss, optimizer=optimizer,
                               metrics=metrics)
        parallel_model.fit(X[train], y_tr, epochs=epochs,
                           batch_size=batch_size, verbose=verbose)
        y_val = to_categorical(y[val])
        scores = parallel_model.evaluate(X[val], y_val, verbose=verbose)
        print("fold %d: %s: %.2f%%" % (i, parallel_model.metrics_names[1],
                                       scores[1]*100))
        i = i+1
        cvscores.append(scores[1] * 100)
    np.savetxt(name + ".csv", cvscores, delimiter=",")
    print(name + " %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


def Dense_NN(X, y, epochs, name, folds=10, verbose=1, batch_size=32,
             optimizer='adam', loss='categorical_crossentropy',
             metrics=['accuracy']):

    y = np.argmax(y, axis=1)
    kfold = StratifiedKFold(n_splits=folds, shuffle=True)
    cvscores = []
    i = 1
    for train, val in kfold.split(X, y):
        # Building model
        model = Sequential()
        model.add(Dense(
            input_dim=X.shape[1],
            units=1000,
            activation="relu"))

        model.add(Dense(
            units=500,
            activation="relu"))

        model.add(Dense(
            units=200,
            activation="relu"))

        model.add(Dense(
            units=512,
            activation="relu"))

        model.add(Dense(
            units=100,
            activation="relu"))

        model.add(Dense(units=3,
                        activation="softmax"))

        y_tr = to_categorical(y[train])

        parallel_model = multi_gpu_model(model, gpus=4)
        parallel_model.compile(loss=loss, optimizer=optimizer,
                               metrics=metrics)
        parallel_model.fit(X[train], y_tr, epochs=epochs,
                           batch_size=batch_size, verbose=verbose)
        y_val = to_categorical(y[val])
        scores = parallel_model.evaluate(X[val], y_val, verbose=verbose)
        print("fold %d: %s: %.2f%%" % (i, parallel_model.metrics_names[1],
                                       scores[1]*100))
        i = i+1
        cvscores.append(scores[1] * 100)
    np.savetxt(name + ".csv", cvscores, delimiter=",")
    print(name + " %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


def CNN2D_32(X, y, epochs, name, folds=10, verbose=1, batch_size=32,
             optimizer='adam', loss='categorical_crossentropy',
             metrics=['accuracy']):

    y = np.argmax(y, axis=1)
    kfold = StratifiedKFold(n_splits=folds, shuffle=True)
    cvscores = []
    i = 1
    for train, val in kfold.split(X, y):
        # Building model
        model = Sequential()
        model.add(Conv2D(64, (2, 2), data_format="channels_first",
                  input_shape=(X.shape[1], X.shape[2], X.shape[3]),
                  strides=[1, 1]))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2D(64, (2, 2), data_format="channels_first",
                  strides=[1, 1]))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # model.add(Conv2D(16, (3, 3), data_format="channels_first",
        #           strides=[1, 1]))
        # model.add(BatchNormalization())
        # model.add(Activation('relu'))

        # model.add(Conv2D(5, (3, 2), data_format="channels_first",
        #           strides=[2, 1]))
        # model.add(BatchNormalization())
        # model.add(Activation('relu'))

        # model.add(Conv2D(5, (3, 2), data_format="channels_first",
        #           strides=[2, 1]))
        # model.add(BatchNormalization())
        # model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        # model.add(Dense(512, activation='relu'))
        # model.add(Dense(512, activation='relu'))
        # model.add(Dense(512, activation='relu'))
        model.add(Dense(2000, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(800, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(200, activation='relu'))
        model.add(Dense(3, activation='softmax'))

        y_tr = to_categorical(y[train])

        parallel_model = multi_gpu_model(model, gpus=4)
        parallel_model.compile(loss=loss, optimizer=optimizer,
                               metrics=metrics)
        parallel_model.fit(X[train], y_tr, epochs=epochs,
                           batch_size=batch_size, verbose=verbose)
        y_val = to_categorical(y[val])
        scores = parallel_model.evaluate(X[val], y_val, verbose=verbose)
        print("fold %d: %s: %.2f%%" % (i, parallel_model.metrics_names[1],
                                       scores[1]*100))
        i = i+1
        cvscores.append(scores[1] * 100)
    np.savetxt(name + ".csv", cvscores, delimiter=",")
    print(name + " %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


def LSTMNN(X, y, epochs, name, folds=10, verbose=1, batch_size=32,
           optimizer='adam', loss='categorical_crossentropy',
           metrics=['accuracy']):

    y = np.argmax(y, axis=1)
    kfold = StratifiedKFold(n_splits=folds, shuffle=True)
    cvscores = []
    i = 1
    for train, val in kfold.split(X, y):
            # Building model
        model = Sequential()

        # model.add(Conv2D(64, (2, 1), data_format="channels_first",
        #           input_shape=(X.shape[1], X.shape[2], X.shape[3]),
        #           strides=[2, 1]))
        # model.add(BatchNormalization())
        # model.add(Activation('relu'))

        # model.add(Conv1D(filters=1, kernel_size=5, strides=10,
        #                  input_shape=(X.shape[1], 1),
        #                  kernel_initializer='uniform',
        #                  name='1-Conv1D'))
        # model.add(LSTM(64,  return_sequences=False))
        model.add(LSTM(64,  return_sequences=False,
                  input_shape=(X.shape[1], X.shape[2])))
        model.add(Dropout(0.3))
        # model.add(LSTM(32))
        model.add(Dense(3, activation='softmax'))
        y_tr = to_categorical(y[train])

        parallel_model = multi_gpu_model(model, gpus=4)
        parallel_model.compile(loss=loss, optimizer=optimizer,
                               metrics=metrics)
        parallel_model.fit(X[train], y_tr, epochs=epochs,
                           batch_size=batch_size, verbose=verbose)
        y_val = to_categorical(y[val])
        scores = parallel_model.evaluate(X[val], y_val, verbose=verbose)
        print("fold %d: %s: %.2f%%" % (i, parallel_model.metrics_names[1],
                                       scores[1]*100))
        i = i+1
        cvscores.append(scores[1] * 100)
    np.savetxt(name + ".csv", cvscores, delimiter=",")
    print(name + " %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


def ConvLSTM(X, y, epochs, name, folds=10, verbose=1, batch_size=32,
             optimizer='adam', loss='categorical_crossentropy',
             metrics=['accuracy']):

    y = np.argmax(y, axis=1)
    kfold = StratifiedKFold(n_splits=folds, shuffle=True)
    cvscores = []
    i = 1
    for train, val in kfold.split(X, y):
        # Building model
        model = Sequential()
        model.add(ConvLSTM2D(64, (1, 2),  return_sequences=False,
                  input_shape=(2701, 5, 5, 1)))
        model.add(Dropout(0.3))
        model.add(Flatten())
        # model.add(LSTM(32))
        model.add(Dense(3, activation='softmax'))

        y_tr = to_categorical(y[train])

        parallel_model = multi_gpu_model(model, gpus=4)
        parallel_model.compile(loss=loss, optimizer=optimizer,
                               metrics=metrics)
        parallel_model.fit(X[train], y_tr, epochs=epochs,
                           batch_size=batch_size, verbose=verbose)
        y_val = to_categorical(y[val])
        scores = parallel_model.evaluate(X[val], y_val, verbose=verbose)
        print("fold %d: %s: %.2f%%" % (i, parallel_model.metrics_names[1],
                                       scores[1]*100))
        i = i+1
        cvscores.append(scores[1] * 100)
    np.savetxt(name + ".csv", cvscores, delimiter=",")
    print(name + " %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
