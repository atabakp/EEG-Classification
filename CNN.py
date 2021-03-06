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


def CNN1D(X, y, epochs, name, train=True, test_split_size=0.1, verbose=1,
          num_GPU=0, batch_size=32, optimizer='adam',
          loss='categorical_crossentropy', metrics=['accuracy'], shuffle=True):

    TBlog_path = ('./TrainedModels/logs/' +
                  name+'-'+datetime.datetime.now()
                  .strftime('%Y-%m-%d_%H-%M-%S'))
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
    # all_callbacks = [tensorboard]

    # Building model
    model = Sequential()
    model.add(Conv1D(filters=1, kernel_size=5, strides=10,
                     input_shape=(X.shape[1], 1), kernel_initializer='uniform',
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
    model.summary()
    # Training
    if num_GPU < 2:  # no or single GPU systems
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        if train:
            model.fit(X, y, batch_size=batch_size, epochs=epochs,
                      verbose=verbose, callbacks=all_callbacks,
                      validation_split=0.1, shuffle=shuffle)
    else:  # prallelized on multiple GPUs
        from keras.utils import multi_gpu_model
        parallel_model = multi_gpu_model(model, gpus=num_GPU)
        parallel_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        if train:
            parallel_model.fit(X, y, batch_size=batch_size*num_GPU,
                               epochs=epochs, verbose=verbose,
                               callbacks=all_callbacks,
                               validation_split=0.1, shuffle=shuffle)
    model.save(Model_save_path+name+'.final.hdf5')
    model_json = model.to_json()
    with open(Model_save_path+name+'.final.json', "w") as json_file:
        json_file.write(model_json)
    return parallel_model


def CNN2D(X, y, epochs, name, test_split_size=0.1, verbose=1,
          num_GPU=0, batch_size=32, optimizer='adam',
          loss='categorical_crossentropy', metrics=['accuracy'], shuffle=True):

    TBlog_path = ('./TrainedModels/logs/' +
                  name+'-'+datetime.datetime.now()
                  .strftime('%Y-%m-%d_%H-%M-%S'))
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
    # all_callbacks = [tensorboard]

    # Building model
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
    model.add(Dense(2000, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.summary()
    # Training
    if num_GPU < 2:  # no or single GPU systems
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        model.fit(X, y, batch_size=batch_size, epochs=epochs,
                  verbose=verbose, callbacks=all_callbacks,
                  validation_split=0.1, shuffle=shuffle)
    else:  # prallelized on multiple GPUs
        from keras.utils import multi_gpu_model
        parallel_model = multi_gpu_model(model, gpus=num_GPU)
        parallel_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        parallel_model.fit(X, y, batch_size=batch_size*num_GPU,
                           epochs=epochs, verbose=verbose,
                           callbacks=all_callbacks,
                           validation_split=0.1, shuffle=shuffle)
    model.save(Model_save_path+name+'.final.hdf5')
    return model


def Dense_NN(X, y, epochs, name, test_split_size=0.1, verbose=1,
             num_GPU=0, batch_size=32, optimizer='adam',
             loss='categorical_crossentropy', metrics=['accuracy'],
             shuffle=True):

    TBlog_path = ('./TrainedModels/logs/' +
                  name+'-'+datetime.datetime.now()
                  .strftime('%Y-%m-%d_%H-%M-%S'))
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
    model.summary()
    # Training
    if num_GPU < 2:  # no or single GPU systems
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        model.fit(X, y, batch_size=batch_size, epochs=epochs,
                  verbose=verbose, callbacks=all_callbacks,
                  validation_split=0.1, shuffle=shuffle)
    else:  # prallelized on multiple GPUs
        from keras.utils import multi_gpu_model
        parallel_model = multi_gpu_model(model, gpus=num_GPU)
        parallel_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        parallel_model.fit(X, y, batch_size=batch_size*num_GPU,
                           epochs=epochs, verbose=verbose,
                           callbacks=all_callbacks,
                           validation_split=0.1, shuffle=shuffle)
    model.save(Model_save_path+name+'.final.hdf5')
    return model


def CNN2D_32(X, y, epochs, name, test_split_size=0.1, verbose=1,
             num_GPU=0, batch_size=32, optimizer='adam',
             loss='categorical_crossentropy', metrics=['accuracy'],
             shuffle=True):

    TBlog_path = ('./TrainedModels/logs/' +
                  name+'-'+datetime.datetime.now()
                  .strftime('%Y-%m-%d_%H-%M-%S'))
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
    # all_callbacks = [tensorboard]

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
    model.summary()
    # Training
    if num_GPU < 2:  # no or single GPU systems
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        model.fit(X, y, batch_size=batch_size, epochs=epochs,
                  verbose=verbose, callbacks=all_callbacks,
                  validation_split=0.1, shuffle=shuffle)
    else:  # prallelized on multiple GPUs
        from keras.utils import multi_gpu_model
        parallel_model = multi_gpu_model(model, gpus=num_GPU)
        parallel_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        parallel_model.fit(X, y, batch_size=batch_size*num_GPU,
                           epochs=epochs, verbose=verbose,
                           callbacks=all_callbacks,
                           validation_split=0.1, shuffle=shuffle)
    model.save(Model_save_path+name+'.final.hdf5')
    return model


def LSTMNN(X, y, epochs, name, test_split_size=0.1, verbose=1,
           num_GPU=0, batch_size=32, optimizer='adam',
           loss='categorical_crossentropy', metrics=['accuracy'],
           shuffle=True):

    TBlog_path = ('./TrainedModels/logs/' +
                  name+'-'+datetime.datetime.now()
                  .strftime('%Y-%m-%d_%H-%M-%S'))
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
    # all_callbacks = [tensorboard]

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
    #model.add(Dropout(0.5))
    #model.add(Dense(512, activation='relu'))
    #model.add(Dense(512, activation='relu'))
    #model.add(LSTM(32))
    model.add(Dense(3, activation='softmax'))
    model.summary()
    # Training
    if num_GPU < 2:  # no or single GPU systems
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        model.fit(X, y, batch_size=batch_size, epochs=epochs,
                  verbose=verbose, callbacks=all_callbacks,
                  validation_split=0.1)
    else:  # prallelized on multiple GPUs
        from keras.utils import multi_gpu_model
        parallel_model = multi_gpu_model(model, gpus=num_GPU)
        parallel_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        parallel_model.fit(X, y, batch_size=batch_size*num_GPU,
                           epochs=epochs, verbose=verbose,
                           callbacks=all_callbacks,
                           validation_split=0.1)
    model.save(Model_save_path+name+'.final.hdf5')
    model_json = model.to_json()
    with open(Model_save_path+name+'.final.json', "w") as json_file:
        json_file.write(model_json)
    return model


def ConvLSTM(X, y, epochs, name, test_split_size=0.1, verbose=1,
             num_GPU=0, batch_size=32, optimizer='adam',
             loss='categorical_crossentropy', metrics=['accuracy'],
             shuffle=True):

    TBlog_path = ('./TrainedModels/logs/' +
                  name+'-'+datetime.datetime.now()
                  .strftime('%Y-%m-%d_%H-%M-%S'))
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
    # all_callbacks = [tensorboard]

    # Building model
    model = Sequential()
    model.add(ConvLSTM2D(64, (1, 2),  return_sequences=False,
              input_shape=(2701, 5, 5, 1)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    # model.add(LSTM(32))
    model.add(Dense(3, activation='softmax'))
    model.summary()
    # Training
    if num_GPU < 2:  # no or single GPU systems
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        model.fit(X, y, batch_size=batch_size, epochs=epochs,
                  verbose=verbose, callbacks=all_callbacks,
                  validation_split=0.1, shuffle=shuffle)
    else:  # prallelized on multiple GPUs
        from keras.utils import multi_gpu_model
        parallel_model = multi_gpu_model(model, gpus=num_GPU)
        parallel_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        parallel_model.fit(X, y, batch_size=batch_size*num_GPU,
                           epochs=epochs, verbose=verbose,
                           callbacks=all_callbacks,
                           validation_split=0.1, shuffle=shuffle)
    model.save(Model_save_path+name+'.final.hdf5')
    model_json = model.to_json()
    with open(Model_save_path+name+'.final.json', "w") as json_file:
        json_file.write(model_json)
    return model
