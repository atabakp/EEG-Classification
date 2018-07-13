#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from scipy import signal
import tqdm


# Wavelet Spectogram
def wave_spec(EEG, width, wavelet=signal.morlet, filename=None):

    widths = np.arange(1, width)
    cwtmatr = np.stack([np.hstack([signal.cwt(EEG[j, :, i], wavelet, widths)
                        for i in range(EEG.shape[2])])
                        for j in tqdm(range(EEG.shape[0]))])
    print("Wavelet shape: ", cwtmatr.shape)
    if filename is None:
        np.save('cwtmatr'+str(width), cwtmatr)
        print("Saved as:", 'cwtmatr'+str(width))
    else:
        np.save(filename, cwtmatr)
        print("Saved as:", filename)


# Short-Time Fourier Transform
def short_time_ft(EEG, fs=100, filename=None):
    sft = np.stack([np.hstack([signal.stft(EEG[j, :, i], fs=fs)[2]
                   for i in range(EEG.shape[2])])
                   for j in tqdm(range(EEG.shape[0]))])
    print("STFT shape: ", sft.shape)
    if filename is None:
        np.save('sft'+str(fs), sft)
        print("Saved as:", 'sft'+str(fs))
    else:
        np.save(filename, sft)
        print("Saved as:", filename)


# Multitaper spectogram
def short_time_ft(EEG, npts=20, fw=3, number_of_tapers=5, fs=100,
                  filename=None):
    tapers, _, _ = mtspec.dpss(npts=20, fw=3, number_of_tapers=5)
    tf = np.stack([np.hstack(
        [np.mean(np.power(np.abs([signal.stft(EEG[j, :, i],
                                  fs=fs, window=tapers[:, t],
                                  nperseg=tapers.shape[0])[2]
                                  for t in range(tapers.shape[1])]), 2), axis=0)
            for i in range(EEG.shape[2])])
                                  for j in tqdm(range(EEG.shape[0]))])
    print("MultiTaper shape: ", tf.shape)
    if filename is None:
        np.save('tf'+str(fs), tf)
        print("Saved as:", 'sft'+str(fs))
    else:
        np.save(filename, tf)
        print("Saved as:", filename)
