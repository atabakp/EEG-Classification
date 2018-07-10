#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import (Dense, Dropout, Conv1D, GlobalAveragePooling1D,
                          MaxPooling1D, Flatten)
from keras import callbacks
import datetime
import os


def CNN1D(X, y, epochs, name, test_split_size=0.1, verbose=1,
          no_GPU=0, batch_size=32, optimizer='adam',
          loss='binary_crossentropy', metrics=['accuracy']):

    X_train, _, y_train, _ = train_test_split(X, y, test_size=.1)

    TBlog_path = ('./TrainedModels/logs/' +
                  name+'-'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    Model_save_path = ('./TrainedModels/model/'+name+'/'+'/' +
                       datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') +
                       '/')
    os.makedirs(Model_save_path)

    # Call backs
    checkpoint = callbacks.ModelCheckpoint(
        filepath=Model_save_path+name+'.{epoch}-{val_loss:.3f}.hdf5',
        monitor='val_loss', verbose=0, save_best_only=True,
        save_weights_only=False, mode='min', period=1)

    reduceLR = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=10, verbose=0,
        mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

    EarlyStop = callbacks.EarlyStopping(
        monitor='loss', min_delta=0, patience=20, verbose=0,
        mode='auto', baseline=None)

    tensorboard = callbacks.TensorBoard(log_dir=TBlog_path)

    all_callbacks = [tensorboard, checkpoint, reduceLR, EarlyStop]
    all_callbacks = [tensorboard]

    # Building model
    model = Sequential()
    model.add(Conv1D(filters=1, kernel_size=5, strides=10,
                     input_shape=(X.shape[1], 1), kernel_initializer='uniform',
                     activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    model.summary()
    # Training
    if no_GPU < 2:  # no or single GPU systems
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                  verbose=verbose, callbacks=all_callbacks,
                  validation_split=0.1)
    else:  # prallelized on multiple GPUs
        from keras.utils import multi_gpu_model
        parallel_model = multi_gpu_model(model, gpus=no_GPU)
        parallel_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        parallel_model.fit(X_train, y_train, batch_size=batch_size*no_GPU,
                           epochs=epochs, verbose=verbose,
                           callbacks=all_callbacks,
                           validation_split=0.1)
    return model


def CNN2D(X, y, epochs, name, test_split_size=0.1, verbose=1,
          no_GPU=0, batch_size=32, optimizer='adam',
          loss='binary_crossentropy', metrics=['accuracy']):

    X_train, _, y_train, _ = train_test_split(X, y, test_size=.1)

    TBlog_path = ('./TrainedModels/logs/' +
                  name+'-'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    Model_save_path = ('./TrainedModels/model/'+name+'/'+'/' +
                       datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') +
                       '/')
    os.makedirs(Model_save_path)

    # Call backs
    checkpoint = callbacks.ModelCheckpoint(
        filepath=Model_save_path+name+'.{epoch}-{val_loss:.3f}.hdf5',
        monitor='val_loss', verbose=0, save_best_only=True,
        save_weights_only=False, mode='min', period=1)

    reduceLR = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=10, verbose=0,
        mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

    EarlyStop = callbacks.EarlyStopping(
        monitor='loss', min_delta=0, patience=20, verbose=0,
        mode='auto', baseline=None)

    tensorboard = callbacks.TensorBoard(log_dir=TBlog_path)

    all_callbacks = [tensorboard, checkpoint, reduceLR, EarlyStop]
    all_callbacks = [tensorboard]

    # Building model

    myInput = layers.Input(shape=(28, 28, 1))
    conv1 = layers.Conv2D(16, 3, activation='relu',
                          padding='same', strides=2)(myInput)
    conv2 = layers.Conv2D(32, 3, activation='relu',
                          padding='same', strides=2)(conv1)
    flat = layers.Flatten()(conv2)
    out_layer = layers.Dense(3, activation='softmax')(flat)
    model = Model(myInput, out_layer)
    model.summary()
    # Training
    if no_GPU < 2:  # no or single GPU systems
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                  verbose=verbose, callbacks=all_callbacks,
                  validation_split=0.1)
    else:  # prallelized on multiple GPUs
        from keras.utils import multi_gpu_model
        parallel_model = multi_gpu_model(model, gpus=no_GPU)
        parallel_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        parallel_model.fit(X_train, y_train, batch_size=batch_size*no_GPU,
                           epochs=epochs, verbose=verbose,
                           callbacks=all_callbacks,
                           validation_split=0.1)
    return model
