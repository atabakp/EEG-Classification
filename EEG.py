#!/usr/bin/env python3
# -*- coding: utf-8 -*-



verbose = 0
epoch32 = 50
epoch200 = 5 
#region###### Read Data ###################
from scipy.io import loadmat
from pywt import wavedec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


tr_eeg = loadmat('trimmedData.EEG.mat')
y = pd.read_csv('label.csv',header=None).values

EEG = tr_eeg['EEGtrimmed']
EEG = np.moveaxis(EEG, -1, 0)
print('EEG shape: ',EEG.shape)
print('Labels shape: ',y.shape)

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
y_cat = np_utils.to_categorical(encoded_Y)



#endregion

#region######### MODEL1 #####################
# X = np.vstack([np.concatenate(
#         [np.concatenate(
#         wavedec(EEG[i,:,j], 'db6', level=6),axis=0) for j in range(EEG.shape[2])],axis=0)
#           for i in range(EEG.shape[0])])
# print('X shape after Wavelet: ',X.shape)
# X = X.reshape(X.shape[0],X.shape[1],1)

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=.1)


# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.layers import Embedding
# from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
# from keras import callbacks

# tensorboard = callbacks.TensorBoard(log_dir='./TB/1')

# model = Sequential()
# model.add(Conv1D(64, 3, activation='relu', input_shape=(X.shape[1], 1)))
# model.add(Conv1D(64, 3, activation='relu'))
# model.add(MaxPooling1D(3))
# model.add(Conv1D(128, 3, activation='relu'))
# model.add(Conv1D(128, 3, activation='relu'))
# model.add(GlobalAveragePooling1D())
# model.add(Dropout(0.5))
# model.add(Dense(3, activation='softmax'))
# model.summary()

# from keras.utils import multi_gpu_model
# from keras.optimizers import SGD
# # Replicates `model` on 4 GPUs.
# # This assumes that your machine has 4 available GPUs.
# parallel_model = multi_gpu_model(model, gpus=4)
# sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)
# parallel_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])

# # This `fit` call will be distributed on 4 GPUs.

# #parallel_model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
# parallel_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# parallel_model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=verbose, callbacks=[tensorboard], shuffle=False, validation_data=(X_test,y_test))
# #model.fit(X_train, y_train, batch_size=10, epochs=1, verbose=verbose)
# print(parallel_model.evaluate(X_test, y_test))
# parallel_model.save('model.h5')


# #parallel_model.fit(X_train, y_train, batch_size=200, epochs=5, verbose=verbose)
# #model.fit(X_train, y_train, batch_size=10, epochs=1, verbose=verbose)
# #print(parallel_model.evaluate(X_test, y_test))
#endregion

#region######### MODEL2 #####################
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.layers import Embedding
# from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, Flatten
# from keras import callbacks

# tensorboard = callbacks.TensorBoard(log_dir="./TB/2")

# model = Sequential()
# model.add(Conv1D(1, 5, activation='relu', input_shape=(X.shape[1], 1)))
# #model.add(Flatten())
# model.add(MaxPooling1D(2))
# model.add(GlobalAveragePooling1D())
# model.add(Dense(512, activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(3, activation='softmax'))
# model.summary()

# from keras.utils import multi_gpu_model
# from keras.optimizers import SGD
# # Replicates `model` on 4 GPUs.
# # This assumes that your machine has 4 available GPUs.
# parallel_model = multi_gpu_model(model, gpus=4)

# sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)
# parallel_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
# #parallel_model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
# #parallel_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# parallel_model.fit(X_train, y_train, batch_size=32, epochs=epoch32, verbose=verbose, callbacks=[tensorboard], shuffle=True, validation_data=(X_test,y_test))
# parallel_model.fit(X_train, y_train, batch_size=200, epochs=epoch200, verbose=verbose)
# print(parallel_model.evaluate(X_test, y_test))
#endregion

#region######### MODEL3 #####################

X = np.array([EEG[i,:,5:10].flatten() for i in range(EEG.shape[0])])
X = X.reshape(X.shape[0],X.shape[1],1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=.1)

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, Flatten
from keras import callbacks

tensorboard = callbacks.TensorBoard(log_dir="./TB/3TimeSgd")

model = Sequential()
model.add(Conv1D(1, 5, activation='relu', input_shape=(X.shape[1], 1)))
#model.add(Flatten())
model.add(MaxPooling1D(2))
model.add(GlobalAveragePooling1D())
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()

from keras.utils import multi_gpu_model
from keras.optimizers import SGD
# Replicates `model` on 4 GPUs.
# This assumes that your machine has 4 available GPUs.
parallel_model = multi_gpu_model(model, gpus=4)

sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)
parallel_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
#parallel_model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
#parallel_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

parallel_model.fit(X_train, y_train, batch_size=32, epochs=epoch32, verbose=verbose, callbacks=[tensorboard])
parallel_model.fit(X_train, y_train, batch_size=200, epochs=epoch200, verbose=verbose)
print(parallel_model.evaluate(X_test, y_test))
#endregion

#region######### MODEL4 #####################

X = np.array([EEG[i,:,5:10].flatten() for i in range(EEG.shape[0])])
xmax, xmin = X.max(), X.min()
X = (X - xmin)/(xmax - xmin)
X.astype('float32')
X = X.reshape(X.shape[0],X.shape[1],1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=.1)

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, Flatten
from keras import callbacks

tensorboard = callbacks.TensorBoard(log_dir="./TB/3Timenormsgd")

model = Sequential()
model.add(Conv1D(1, 5, activation='relu', input_shape=(X.shape[1], 1)))
#model.add(Flatten())
model.add(MaxPooling1D(2))
model.add(GlobalAveragePooling1D())
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()


from keras.utils import multi_gpu_model
from keras.optimizers import SGD
# Replicates `model` on 4 GPUs.
# This assumes that your machine has 4 available GPUs.
parallel_model = multi_gpu_model(model, gpus=4)

sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)
parallel_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
#parallel_model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
#parallel_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

parallel_model.fit(X_train, y_train, batch_size=32, epochs=epoch32, verbose=verbose, callbacks=[tensorboard])
parallel_model.fit(X_train, y_train, batch_size=200, epochs=epoch200, verbose=verbose)
print(parallel_model.evaluate(X_test, y_test))
#endregion

#region######### MODEL5 #####################

X = np.array([EEG[i,:,5:10].flatten() for i in range(EEG.shape[0])])
X = X.reshape(X.shape[0],X.shape[1],1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=.1)

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, Flatten
from keras import callbacks

tensorboard = callbacks.TensorBoard(log_dir="./TB/3Timeadam")

model = Sequential()
model.add(Conv1D(1, 5, activation='relu', input_shape=(X.shape[1], 1)))
#model.add(Flatten())
model.add(MaxPooling1D(2))
model.add(GlobalAveragePooling1D())
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()

from keras.utils import multi_gpu_model
from keras.optimizers import SGD
# Replicates `model` on 4 GPUs.
# This assumes that your machine has 4 available GPUs.
parallel_model = multi_gpu_model(model, gpus=4)

sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)
#parallel_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
#parallel_model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
parallel_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

parallel_model.fit(X_train, y_train, batch_size=32, epochs=epoch32, verbose=verbose, callbacks=[tensorboard])
parallel_model.fit(X_train, y_train, batch_size=200, epochs=epoch200, verbose=verbose)
print(parallel_model.evaluate(X_test, y_test))
#endregion

#region######### MODEL6 #####################


X = np.array([EEG[i,:,5:10].flatten() for i in range(EEG.shape[0])])
xmax, xmin = X.max(), X.min()
X = (X - xmin)/(xmax - xmin)
X.astype('float32')
X = X.reshape(X.shape[0],X.shape[1],1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=.1)

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, Flatten
from keras import callbacks

tensorboard = callbacks.TensorBoard(log_dir="./TB/3Timenormadam")

model = Sequential()
model.add(Conv1D(1, 5, activation='relu', input_shape=(X.shape[1], 1)))
#model.add(Flatten())
model.add(MaxPooling1D(2))
model.add(GlobalAveragePooling1D())
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()


from keras.utils import multi_gpu_model
from keras.optimizers import SGD
# Replicates `model` on 4 GPUs.
# This assumes that your machine has 4 available GPUs.
parallel_model = multi_gpu_model(model, gpus=4)

sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)
#parallel_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
#parallel_model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
parallel_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

parallel_model.fit(X_train, y_train, batch_size=32, epochs=epoch32, verbose=verbose, callbacks=[tensorboard])
parallel_model.fit(X_train, y_train, batch_size=200, epochs=epoch200, verbose=verbose)
print(parallel_model.evaluate(X_test, y_test))
#endregion

#region######### MODEL7 #####################
X = np.vstack([np.concatenate(
        [np.concatenate(
        wavedec(EEG[i,:,j], 'db6', level=6),axis=0) for j in range(EEG.shape[2])],axis=0)
          for i in range(EEG.shape[0])])
print('X shape after Wavelet: ',X.shape)
X = X.reshape(X.shape[0],X.shape[1],1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=.1)

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras import callbacks

tensorboard = callbacks.TensorBoard(log_dir="./TB/3Timenormadam")

model = Sequential()
model.add(Conv1D(1, 5, activation='relu', input_shape=(X.shape[1], 1)))
#model.add(Flatten())
model.add(MaxPooling1D(2))
model.add(GlobalAveragePooling1D())
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()


from keras.utils import multi_gpu_model
from keras.optimizers import SGD
# Replicates `model` on 4 GPUs.
# This assumes that your machine has 4 available GPUs.
parallel_model = multi_gpu_model(model, gpus=4)

sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)
#parallel_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
#parallel_model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
parallel_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

parallel_model.fit(X_train, y_train, batch_size=32, epochs=epoch32, verbose=verbose, callbacks=[tensorboard])
parallel_model.fit(X_train, y_train, batch_size=200, epochs=epoch200, verbose=verbose)
print(parallel_model.evaluate(X_test, y_test))
#endregion
