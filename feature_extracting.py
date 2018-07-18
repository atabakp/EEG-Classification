#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy import signal
import tqdm
import mtspec


# Wavelet Spectogram
def wave_spec(EEG, width=5, wavelet=signal.morlet, filename=None):

    widths = np.arange(1, width)
    cwtmatr = np.stack([np.hstack([signal.cwt(EEG[j, :, i], wavelet, widths)
                        for i in range(EEG.shape[2])])
                        for j in range(EEG.shape[0])])
    print("Wavelet shape: ", cwtmatr.shape)
    if filename is None:
        np.save('cwtmatr'+str(width), cwtmatr)
        print("Saved as:", 'cwtmatr'+str(width))
    else:
        np.save(filename, cwtmatr)
        print("Saved as:", filename)


# Wavelet Spectogram 2D
def wave_spec_2D(EEG, width=5, wavelet=signal.morlet, filename=None):

    widths = np.arange(1, width)
    cwtmatr = np.stack([signal.cwt(EEG[j, :, i], signal.morlet, widths)
                       for i in range(EEG.shape[2])]
                       for j in tqdm(range(EEG.shape[0])))
    print("Wavelet shape: ", cwtmatr.shape)
    if filename is None:
        np.save('cwtmatr_2D'+str(width), cwtmatr)
        print("Saved as:", 'cwtmatr_2D'+str(width))
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


# Short-Time Fourier Transform 2D
def short_time_ft_2D(EEG, fs=100, filename=None):
    sft = np.stack([signal.stft(EEG[j, :, i], fs=100, nperseg=40)[2]
                   for i in range(EEG.shape[2])]
                   for j in tqdm(range(EEG.shape[0])))
    print("STFT shape: ", sft.shape)
    if filename is None:
        np.save('sft_2D'+str(fs), sft)
        print("Saved as:", 'sft_2D'+str(fs))
    else:
        np.save(filename, sft)
        print("Saved as:", filename)
    return sft


# Multitaper spectogram
def multitaper(EEG, npts=20, fw=3, number_of_tapers=5, fs=100,
               filename=None):
    tapers, _, _ = mtspec.dpss(npts=20, fw=3, number_of_tapers=5)
    tf = np.stack([np.hstack(
        [np.mean(np.power(np.abs([signal.stft(EEG[j, :, i],
                                  fs=fs, window=tapers[:, t],
                                  nperseg=tapers.shape[0])[2]
                                  for t in range(tapers.shape[1])]), 2),
                 axis=0)
            for i in range(EEG.shape[2])])
                                  for j in tqdm(range(EEG.shape[0]))])
    print("MultiTaper shape: ", tf.shape)
    if filename is None:
        np.save('tf'+str(fs), tf)
        print("Saved as:", 'tf'+str(fs))
    else:
        np.save(filename, tf)
        print("Saved as:", filename)


# Multitaper spectogram 2D
def multitaper_2D(EEG, npts=20, fw=3, number_of_tapers=5, fs=100,
                  filename=None):
    tapers, _, _ = mtspec.dpss(npts=20, fw=3, number_of_tapers=5)
    tf = np.stack(
        [np.mean(np.power(np.abs([signal.stft(EEG[j, :, i],
                                 fs=100, window=tapers[:, t],
                                 nperseg=tapers.shape[0])[2]
                                 for t in range(tapers.shape[1])]), 2), axis=0)
         for i in range(EEG.shape[2])] for j in tqdm(range(EEG.shape[0])))
    print("MultiTaper shape: ", tf.shape)
    if filename is None:
        np.save('tf_2D'+str(fs), tf)
        print("Saved as:", 'tf_2D'+str(fs))
    else:
        np.save(filename, tf)
        print("Saved as:", filename)


def main():
    EEG = np.load('EEG.npy')
    # Wavelet Spectogram concatinate
    print('Wavelet 1D width=50...')
    wave_spec(EEG, width=5, wavelet=signal.morlet, filename='cwt-1D-5')

    print('Wavelet 2D width=50...')
    # Wavelet Spectogram 2D
    wave_spec_2D(EEG, width=5, wavelet=signal.morlet, filename='cwt-1D-5')

    # Short-Time Fourier Transform
    print('SFTF 1D fs=50...')
    short_time_ft(EEG, fs=50, filename='stft-1D-50')
    print('SFTF 1D fs=100...')
    short_time_ft(EEG, fs=100, filename='stft-1D-100')
    print('SFTF 1D fs=150...')
    short_time_ft(EEG, fs=150, filename='stft-1D-150')

    # Short-Time Fourier Transform 2D
    print('SFTF 2D fs=50...')
    short_time_ft_2D(EEG, fs=50, filename='stft-2D-50')
    print('SFTF 2D fs=100...')
    short_time_ft_2D(EEG, fs=100, filename='stft-2D-100')
    print('SFTF 2D fs=150...')
    short_time_ft_2D(EEG, fs=150, filename='stft-2D-150')

    # Multitaper spectogram
    print('multitaper 1D fs=50...')
    multitaper(EEG, npts=20, fw=3, number_of_tapers=5, fs=50,
               filename='mt-1D-50')

    print('multitaper 1D fs=100...')
    multitaper(EEG, npts=20, fw=3, number_of_tapers=5, fs=100,
               filename='mt-1D-100')

    print('multitaper 1D fs=150...')
    multitaper(EEG, npts=20, fw=3, number_of_tapers=5, fs=150,
               filename='mt-1D-150')

    # Multitaper spectogram 2D
    print('multitaper 2D fs=50...')
    multitaper_2D(EEG, npts=20, fw=3, number_of_tapers=5, fs=50,
                  filename='mt-2D-50')
    print('multitaper 2D fs=100...')             
    multitaper_2D(EEG, npts=20, fw=3, number_of_tapers=5, fs=100,
                  filename='mt-2D-100')
    print('multitaper 2D fs=150...')             
    multitaper_2D(EEG, npts=20, fw=3, number_of_tapers=5, fs=150,
                  filename='mt-2D-150')

if __name__ == "__main__":
    main()
