# Author: Adam Fong
# Date: December 27, 2021
# Purpose: Create a function that takes a wave file and preprocesses it to be valid input for angle grinder tflite model
import math
import numpy as np
import pandas as pd
from pickle import load
import scipy.io.wavfile as wav
import sklearn

samplerate = 44100
N_BINS = 1000

# making chunk size
# currently only neat if each chunk is 1 second 
def createChunks(raw_data_array):
    seconds = 2
    chunk_size = int(seconds * samplerate)
    chunks_final = pd.DataFrame([np.zeros(chunk_size)])

    # if seconds != 1, there will be some lost data. Hard to avoid this if we are going to have a lot of time recorded 
    # removes time from the beginning of the recording because there is more often noise there than at the end
    for rec in raw_data_array:
        n_chunks = math.floor(len(rec) / chunk_size)
        for i in range(len(rec) - (n_chunks * chunk_size), len(rec), chunk_size):
            chunk = rec[i:(i+chunk_size)]
            #print(f"Length: {len(chunk)}, First Value: {chunk[0]}, Last Value: {chunk[len(chunk) - 1]}")
            chunks_final = chunks_final.append(pd.Series(chunk), ignore_index = True)

    # get rid of filler zero's line
    chunks_final = chunks_final.iloc[1:, :]
    return chunks_final

def rawAudioToFreq(arr: np.array, bins: int):
    n = len(arr)                       # length of the signal
    k = np.arange(n)
    T = n/samplerate
    
    frq = k/T # two sides frequency range
    
    
    zz=int(n/2)
    freq = frq[range(zz)]           # one side frequency range
    Y0 = np.fft.fft(arr)/n              # fft computing and normalization
    Y = Y0[range(zz)]

    # obtaining maximum amplitude and its corresponding frequency 
    Y_max = abs(Y).max()
    freq_max = freq[np.where(abs(Y) == Y_max)[0][0]]
    
    arr = np.array([freq.astype(int), Y.astype(int)])
    bin_size = math.floor(arr[0, arr.shape[1]-1] / bins)
    bin_minimums = np.arange(0, arr[0, arr.shape[1] - 1], bin_size)
    bin_arr = np.array([bin_minimums, np.zeros(len(bin_minimums))])

    # collecting magintudes in bins 
    for i in range(0, arr[0, arr.shape[1] - 1], math.floor(bin_size)):
        bin_arr[1, int(i / bin_size)] = np.sum(abs(Y)[i:(i+bin_size)])
    
    #plt.plot(freq, abs(Y))
    #plt.xlim([freq_max - 100, freq_max + 100])
    return freq, abs(Y), bin_arr, Y_max, freq_max

def getFreqs(chunk_df: pd.DataFrame, bins: int):
    freqs_df = pd.DataFrame(np.zeros(bins))
    for row in range(chunk_df.shape[0]):
        
        freqs, Ys, bin_array, Y_max, freq_max = rawAudioToFreq(chunk_df.to_numpy()[row, :], bins)

        if row == 0:
            freqs_df = pd.DataFrame([bin_array[0, :]])
        else:
            freqs_df = freqs_df.append(pd.Series(bin_array[1,:]), ignore_index = True)
            
    return freqs_df

def audioProcessor(file):
    # loading the same scaler that scaled the data for model training 
    scaler = load(open('C:/Users/adamf/OneDrive/Documents/university/UBC/homework_Winter_2021/IGEN_330/BikeSentry/device/audio_recognition/fft_examples/audio_scaler.pkl', 'rb'))

    # take file and convert to pandas dataframe
    sr, y = wav.read(file)

    # time of clips
    s = 2 #seconds

    # force file to exactly 2 seconds or 44.1k * 2sec samples
    y = y[:sr*s]

    # create chunk
    chunk = createChunks(y)

    # get frequencies
    freqs = getFreqs(chunk, N_BINS)

    # scale frequencies
    freqs_scaled = scaler.fit_transform(freqs.to_numpy()[1:, :]) #first row is bin values

    return freqs_scaled

if __name__ == "__main__":
    filename = "C:/Users/adamf/OneDrive/Documents/university/UBC/homework_Winter_2021/IGEN_330/BikeSentry_data/angle-grinders/blue-1m-trimmed.wav"
    input = audioProcessor(filename)
    
    print(input)