#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

# # Read from .mat file
# tr_eeg = loadmat('trimmedData.EEG.mat')
# y = pd.read_csv('label.csv', header=None).values
# EEG = tr_eeg['EEGtrimmed']
# EEG = np.moveaxis(EEG, -1, 0)
# np.save('EEG', EEG)

# # # convert integers to dummy variables (i.e. one hot encoded)
# encoder = LabelEncoder()
# encoder.fit(y)
# encoded_Y = encoder.transform(y)
# y = np_utils.to_categorical(encoded_Y)
# np.save('y', y)

EEG = np.load('EEG.npy')
temp_list = []

for i in range(EEG.shape[0]):
    f7 = EEG[i, :, 2]
    f3 = EEG[i, :, 3]
    fz = EEG[i, :, 4]
    f4 = EEG[i, :, 5]
    f8 = EEG[i, :, 6]
    fc5 = EEG[i, :, 7]
    fc1 = EEG[i, :, 8]
    fc2 = EEG[i, :, 9]
    fc6 = EEG[i, :, 10]
    t7 = EEG[i, :, 11]
    c3 = EEG[i, :, 12]
    cz = EEG[i, :, 13]
    c4 = EEG[i, :, 14]
    t8 = EEG[i, :, 15]
    tp9 = EEG[i, :, 16]
    cp5 = EEG[i, :, 17]
    cp1 = EEG[i, :, 18]
    cp2 = EEG[i, :, 19]
    cp6 = EEG[i, :, 20]
    p7 = EEG[i, :, 22]
    p3 = EEG[i, :, 23]
    pz = EEG[i, :, 24]
    p4 = EEG[i, :, 25]
    p8 = EEG[i, :, 26]
    new = np.zeros((5, 5, 2701))
    new[0, 0] = f7
    new[0, 1] = f3
    new[0, 2] = fz
    new[0, 3] = f4
    new[0, 4] = f8
    new[1, 0] = 0
    new[1, 1] = fc5
    new[1, 2] = fc1
    new[1, 3] = fc2
    new[1, 4] = fc6
    new[2, 0] = t7
    new[2, 1] = c3
    new[2, 2] = cz
    new[2, 3] = c4
    new[2, 4] = t8
    new[3, 0] = tp9
    new[3, 1] = cp5
    new[3, 2] = cp1
    new[3, 3] = cp2
    new[3, 4] = cp6
    new[4, 0] = p7
    new[4, 1] = p3
    new[4, 2] = pz
    new[4, 3] = p4
    new[4, 4] = p8
    temp_list.append(new)

EEG_2D = np.stack(temp_list)
np.save('EEG_2D', EEG_2D)
