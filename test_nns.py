#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import CNN
import talos as ta



import numpy as np
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


num_GPU = 4
epochs = 100
verbose = 0
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


def CNN2D_grid(x_train, y_train, x_val, y_val, params):
    test_split_size = 0.1
    shuffle = False
    name = 'grid'
    X_train, _, y_train, _ = train_test_split(X, y, test_size=test_split_size)

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
    # model.add(Dense(512, activation='relu'))
    model.add(Dense(2000, activation='relu'))
    model.add(Dropout(params['dropout']))
    model.add(Dense(800, activation='relu'))
    model.add(Dropout(params['dropout']))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.summary()
    # Training
    if 4 < 2:  # no or single GPU systems
        model.compile(loss=params['losses'], optimizer=params['optimizer'], metrics=['accuracy'])
        history = model.fit(X_train, y_train, batch_size=params['batch_size'], epochs=100,
                  verbose=0, callbacks=all_callbacks,
                  validation_split=0.1, shuffle=shuffle)
    else:  # prallelized on multiple GPUs
        from keras.utils import multi_gpu_model
        parallel_model = multi_gpu_model(model, gpus=4)
        parallel_model.compile(loss=params['losses'], optimizer=params['optimizer'], metrics=['accuracy'])
        history = parallel_model.fit(X_train, y_train, batch_size=params['batch_size'],
                           epochs=100, verbose=0,
                           callbacks=all_callbacks,
                           validation_split=0.1, shuffle=shuffle)
    model.save(Model_save_path+name+'.final.hdf5')
    return history, model

p = {'lr': (0.5, 5, 10),
     'first_neuron':[4, 8, 16, 32, 64],
     'hidden_layers':[0, 1, 2],
     'batch_size': (2, 30, 10),
     'epochs': [150],
     'dropout': (0, 0.5, 5),
     'weight_regulizer':[None],
     'emb_output_dims': [None],
     'shape':['brick','long_funnel'],
     'optimizer': ['adam', 'Nadam', 'RMSprop'],
     'losses': ['logcosh', 'binary_crossentropy'],
     'activation':['relu', 'elu'],
     'last_activation': ['sigmoid']}
y = np.load('y.npy')
X = np.abs(np.load('stft-2D-100.npy'))
print('start')
t = ta.Scan(x=X,
            y=y,
            model=CNN2D_grid,
            grid_downsample=0.01, 
            params=p,
            dataset_name='EEG',
            experiment_no='1')



# Read data
y = np.load('y.npy')
# ################ TRAIN MODELS #########################

# LSTM
X = np.load('EEG.npy')
CNN.LSTMNN(X, y, epochs=epochs, name='LSTM', num_GPU=num_GPU,
           optimizer='adam', batch_size=32, loss='binary_crossentropy',
           metrics=['accuracy'], test_split_size=0.1, verbose=1)
Print("Done!!!!!!!!!!!!!!!!!")

# channels all
X = np.load('EEG.npy')
X = reshape_1D_conv(X)
CNN.CNN1D(X, y, epochs=epochs, name='TimeDomain-allch', num_GPU=num_GPU,
          optimizer='adam', batch_size=32, loss='binary_crossentropy',
          metrics=['accuracy'], test_split_size=0.1, verbose=verbose)


# channels 7:21
X = np.load('EEG.npy')
X = reshape_1D_conv(X[:, :, 7:21])
CNN.CNN1D(X, y, epochs=epochs, name='TimeDomain-ch0:22', num_GPU=num_GPU,
          optimizer='adam', batch_size=32, loss='binary_crossentropy',
          metrics=['accuracy'], test_split_size=0.1, verbose=verbose)

print("Short Time Fourier 1D")
# Short Time Fourier 1D
X = np.abs(np.load('stft-1D-50.npy'))
X = reshape_1D_conv(X)
model = CNN.CNN1D(X, y, epochs=epochs, name='stft-1D-50', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1, verbose=verbose)

X = np.abs(np.load('stft-1D-100.npy'))
X = reshape_1D_conv(X)
model = CNN.CNN1D(X, y, epochs=epochs, name='stft-1D-100', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1, verbose=verbose)

X = np.abs(np.load('stft-1D-150.npy'))
X = reshape_1D_conv(X)
model = CNN.CNN1D(X, y, epochs=epochs, name='stft-1D-150', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1, verbose=verbose)


# Short Time Fourier 1D with log
print("Short Time Fourier 1D with log")
X = np.abs(np.load('stft-1D-50.npy'))
X = np.log10(X.clip(min=000000.1))
X = reshape_1D_conv(X)
model = CNN.CNN1D(X, y, epochs=epochs, name='stft-1D-50-log', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1, verbose=verbose)

X = np.abs(np.load('stft-1D-100.npy'))
X = np.log10(X.clip(min=000000.1))
X = reshape_1D_conv(X)
model = CNN.CNN1D(X, y, epochs=epochs, name='stft-1D-100-log', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1, verbose=verbose)

X = np.abs(np.load('stft-1D-150.npy'))
X = np.log10(X.clip(min=000000.1))
X = reshape_1D_conv(X)
model = CNN.CNN1D(X, y, epochs=epochs, name='stft-1D-150-log', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1, verbose=verbose)


# Short Time Fourier 2D
print("Short Time Fourier 2D")
X = np.abs(np.load('stft-2D-50.npy'))
model = CNN.CNN2D(X, y, epochs=epochs, name='stft-2D-50', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1, verbose=verbose)

X = np.abs(np.load('stft-2D-100.npy'))
model = CNN.CNN2D(X, y, epochs=epochs, name='stft-2D-100', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1, verbose=verbose)

X = np.abs(np.load('stft-2D-100.npy'))
model = CNN.CNN2D(X, y, epochs=epochs, name='stft-2D-150', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1, verbose=verbose)

# STFT 2D with Log Transformation
X = np.abs(np.load('stft-2D-50.npy'))
X = np.log10(X.clip(min=000000.1))
model = CNN.CNN2D(X, y, epochs=epochs, name='stft-2D-50-log', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1, verbose=verbose)

X = np.abs(np.load('stft-2D-100.npy'))
X = np.log10(X.clip(min=000000.1))
model = CNN.CNN2D(X, y, epochs=epochs, name='stft-2D-100-log', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1, verbose=verbose)

X = np.abs(np.load('stft-2D-150.npy'))
X = np.log10(X.clip(min=000000.1))
model = CNN.CNN2D(X, y, epochs=epochs, name='stft-2D-150-log', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1, verbose=verbose)


# Multitaper 1D
print("Multitaper 1D")
X = np.abs(np.load('mt-1D-50.npy'))
X = reshape_1D_conv(X)
model = CNN.CNN1D(X, y, epochs=epochs, name='mt-1D-50', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1, verbose=verbose)

X = np.abs(np.load('mt-1D-100.npy'))
X = reshape_1D_conv(X)
model = CNN.CNN1D(X, y, epochs=epochs, name='mt-1D-100', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1, verbose=verbose)

X = np.abs(np.load('mt-1D-150.npy'))
X = reshape_1D_conv(X)
model = CNN.CNN1D(X, y, epochs=epochs, name='mt-1D-150', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1, verbose=verbose)

# MultiTaper 1D with log
X = np.abs(np.load('mt-1D-50.npy'))
X = np.log10(X.clip(min=000000.1))
X = reshape_1D_conv(X)
model = CNN.CNN1D(X, y, epochs=epochs, name='mt-1D-50-log', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1, verbose=verbose)

X = np.abs(np.load('mt-1D-100.npy'))
X = np.log10(X.clip(min=000000.1))
X = reshape_1D_conv(X)
model = CNN.CNN1D(X, y, epochs=epochs, name='mt-1D-100-log', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1, verbose=verbose)

X = np.abs(np.load('mt-1D-150.npy'))
X = np.log10(X.clip(min=000000.1))
X = reshape_1D_conv(X)
model = CNN.CNN1D(X, y, epochs=epochs, name='mt-1D-150-log', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1, verbose=verbose)


# Multitaper 2D
X = np.abs(np.load('mt-2D-50.npy'))
model = CNN.CNN2D(X, y, epochs=epochs, name='mt-2D-50', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1, verbose=verbose)

X = np.abs(np.load('mt-2D-100.npy'))
model = CNN.CNN2D(X, y, epochs=epochs, name='mt-1D-100', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1, verbose=verbose)

X = np.abs(np.load('mt-2D-150.npy'))
model = CNN.CNN2D(X, y, epochs=epochs, name='mt-1D-150', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1, verbose=verbose)

# Multitaper 2D with Log
X = np.abs(np.load('stft-2D-50.npy'))
X = np.log10(X.clip(min=000000.1))
model = CNN.CNN2D(X, y, epochs=epochs, name='stft-2D-50-log', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1, verbose=verbose)

X = np.abs(np.load('stft-2D-100.npy'))
X = np.log10(X.clip(min=000000.1))
model = CNN.CNN2D(X, y, epochs=epochs, name='stft-2D-100-log', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1, verbose=verbose)

X = np.abs(np.load('stft-2D-150.npy'))
X = np.log10(X.clip(min=000000.1))
model = CNN.CNN2D(X, y, epochs=epochs, name='stft-2D-150-log', num_GPU=num_GPU,
                  optimizer='adam', batch_size=32, loss='binary_crossentropy',
                  metrics=['accuracy'], test_split_size=0.1, verbose=verbose)


# wavelet
# X = np.abs(np.load('cwt-1D-5.npy'))
# X = np.log10(X.clip(min=000000.1))
# X = reshape_1D_conv(X)
# model = CNN.CNN1D(X, y, epochs=epochs, name='cwt-1D-5-log', num_GPU=num_GPU,
#                   optimizer='adam', batch_size=32, loss='binary_crossentropy',
#                   metrics=['accuracy'], test_split_size=0.1, verbose=verbose)

# X = np.abs(np.load('cwt-2D-5.npy'))
# model = CNN.CNN2D(X, y, epochs=epochs, name='cwt-2D-5', num_GPU=num_GPU,
#                   optimizer='adam', batch_size=32, loss='binary_crossentropy',
#                   metrics=['accuracy'], test_split_size=0.1, verbose=verbose)
