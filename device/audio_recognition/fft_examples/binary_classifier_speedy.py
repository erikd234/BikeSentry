# January 19 2022
# Written by Adam Fong
# Changing preprocessing of .wav files to make deployment of audio classifier more realistic

# Plan: See how long fft takes, then see how long binning takes, if my intuition that 
# binning is the limiting factor, refactor how binning works. If it's fft, can't really do anything about that 
#%%
import scipy.io.wavfile as wav
from scipy import stats
import numpy as np
import os
import math
from matplotlib import pyplot as plt
import tensorflow as tf # attempting using tensorflow 2.5.0 to match the tflite_runtime tensorflow version
from pickle import dump
import sklearn 
import time 


def load_data():
    # the 'trimmed' data were manually truncated to the valueable information
    raw_angle_grinders, raw_environ = [], []
    start = time.time()

    for root, dirs, files in os.walk("C:/Users/adamf/OneDrive/Documents/university/UBC/homework_Winter_2021/Term 2/IGEN_330/BikeSentry_data/angle-grinders/"):
        for file in files:  
            # samplerate is constant from the same recording device. If not iPhone XR, do not do this!!!!!
            if("trimmed" in file):
                samplerate, y = wav.read(root + file)
                y0 = y[:, 0]
                y1 = y[:, 1]
                raw_angle_grinders.append(y0)
                raw_angle_grinders.append(y1)
        
            if("envi" in file):
                samplerate, y = wav.read(root + file)
                y0 = y[:, 0]
                y1 = y[:, 1]
                raw_environ.append(y0)
                raw_environ.append(y1)
    
    # see how much data we are processing to make runtimes relative
    total_recorded_data = 0
    for rec in raw_angle_grinders:
        total_recorded_data += len(rec) / samplerate

    for rec in raw_environ:
        total_recorded_data += len(rec) / samplerate

    print(f"Total Amount of Time of Audio Snippets: {total_recorded_data}")
    end = time.time()
    print(f"Runtime of load_data is: {end - start}")

    return raw_angle_grinders, raw_environ, samplerate

def floor_recordings(data_grinders, data_environ, sample_rate):
    # cutting data from the front of the recording to make records to be integer seconds
    start = time.time()

    for index, rec in enumerate(data_grinders):
        s = len(rec) / sample_rate
        floor_s = math.floor(s)
        time_cutoff = s - floor_s
        samples_cutoff = int(time_cutoff * sample_rate)
        data_grinders[index] = rec[samples_cutoff:]
        
        #print(len(data_grinders[index]) / sample_rate)
        

    # repeat for environment
    # cutting data from the front of the recording to make records to be integer seconds
    for index, rec in enumerate(data_environ):
        s = len(rec) / sample_rate
        floor_s = math.floor(s)
        time_cutoff = s - floor_s
        samples_cutoff = int(time_cutoff * sample_rate)
        data_environ[index] = rec[samples_cutoff:]
        
        #print(len(data_environ[index]) / sample_rate)
        
    end = time.time()

    print(f"Runtime of floor_recordings is: {end - start}")

def create_chunks(raw_data, samplerate, seconds):
    start = time.time()
    # making chunks 
    # currently only neat if chunk is 1 second 

    # make empty np.arrays to be populated 
    chunk_size = int(seconds * samplerate)

    # find how many chunks there are in this list 
    n_chunks_total = 0
    for rec in raw_data:
        n_chunks_total += int(len(rec) / chunk_size)
    
    # matrix of n_chunks rows, and chunk_size cols
    chunks_final = np.zeros([n_chunks_total, chunk_size])

    # if seconds != 1, there will be some lost data. Hard to avoid this if we are going to have a lot of time recorded 
    # removes time from the beginning of the recording because there is more often noise there than at the end
    idx = 0

    # populating chunks_final
    for rec in raw_data:
        n_chunks_rec = math.floor(len(rec) / chunk_size)
        for i in range(len(rec) - (n_chunks_rec * chunk_size), len(rec), chunk_size):
            chunk = rec[i:(i+chunk_size)]
            chunks_final[idx] = chunk
            idx += 1

    end = time.time()
    print(f"Runtime of create_chunks is: {end - start}")
    return chunks_final

def raw_audio_to_freq(chunks, samplerate):
    start = time.time()

    chunks_Y = []
    chunks_freqs = []
    for chunk in chunks:
        n = len(chunk) # length of the signal
        k = np.arange(n)
        T = n/samplerate
        
        frq = k/T # two sides frequency range
        
        zz=int(n/2)
        freq = frq[range(zz)]  # one side frequency range
        Y0 = np.fft.fft(chunk)/n  # fft computing and normalization
        Y = Y0[range(zz)]
        chunks_Y.append(abs(Y))
        chunks_freqs.append(freq)
    #plt.plot(freq, abs(Y))
    #plt.xlim([freq_max - 100, freq_max + 100])
    
    end = time.time()
    print(f"Runtime of raw_audio_to_freq is: {end - start}")
    return chunks_Y, chunks_freqs

# without binning, there would be features on the scale of tens of thousands
# may need to 
def bin_chunks_Y(chunks_Y, bins):
    start = time.time()
    chunks_Y_binned = []
    chunks_bin_edges = []
    # sums values within bin edges 
    # TODO: summing may not be the best summary of the data. Max Y or Min Y or a combo of these may catch more variability
    for chunk in chunks_Y:
        chunk_binned = np.array_split(chunk, bins)

        # replace at each index the sum of all the values at the index
        for idx, bin in enumerate(chunk_binned):
            bin_summed = np.sum(bin)
            chunk_binned[idx] = bin_summed
        
        chunks_Y_binned.append(chunk_binned)

    end = time.time()
    print(f"Runtime of bin_chunks_y is: {end - start}")
    return chunks_Y_binned

if __name__ == "__main__":
    #### split audio data into chunks ####

    total_start = time.time()

    raw_data, raw_data_environment, sample_rate = load_data()
    floor_recordings(raw_data, raw_data_environment, sample_rate)
    SECONDS = 1
    chunks_grinder = create_chunks(raw_data, sample_rate, SECONDS)
    chunks_env = create_chunks(raw_data_environment, sample_rate, SECONDS)

    # Y is a list whose values are individual chunks and columns are frequencies
    chunks_grinder_Y, freqs_grinder  = raw_audio_to_freq(chunks_grinder, sample_rate)
    chunks_env_Y, freqs_env = raw_audio_to_freq(chunks_env, sample_rate)

    BINS = 1000
    chunks_grinder_Y_binned = bin_chunks_Y(chunks_grinder_Y, BINS)
    chunks_env_Y_binned = bin_chunks_Y(chunks_env_Y, BINS)

    total_end = time.time()

    print(f"Runtime of total program is: {total_end - total_start}")
    print("done")

# %%
