#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, GlobalAveragePooling1D, MaxPooling1D, Flatten
from keras import callbacks
import datetime


def model(X, y, epochs, test_split_size=0.1, TensorBoard_dir='./logs/1', verbose=1, no_GPU=0, batch_size = 32 ):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)
    tensorboard = callbacks.TensorBoard(log_dir=TensorBoard_dir+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) 
    model = Sequential()
    model.add(Conv1D(filters=1, kernel_size=5 ,strides=10,     
                     input_shape=(X.shape[1],1),kernel_initializer= 'uniform',      
                     activation= 'relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    model.summary()
    if no_GPU < 2:
        model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=[tensorboard])
    else:
        from keras.utils import multi_gpu_model
        parallel_model = multi_gpu_model(model, gpus=no_GPU) 
        parallel_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
        parallel_model.fit(X_train, y_train, batch_size=batch_size*no_GPU, epochs=epochs, verbose=verbose, callbacks=[tensorboard])