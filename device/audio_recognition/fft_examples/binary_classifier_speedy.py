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
import time 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import librosa 


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

def inject_noise(chunk, noise_factor): 

    noise = np.random.randn(len(chunk))
    augmented_chunk = chunk + noise_factor * noise
    # Cast back to same chunk type
    augmented_chunk = augmented_chunk.astype(type(chunk[0]))

    return augmented_chunk
    

def shift_time(chunk, sampling_rate, shift_max, shift_direction):
    shift = np.random.randint(sampling_rate * shift_max)
    if shift_direction == 'right':
        shift = -shift
    elif shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift    

    augmented_chunk = np.roll(chunk, shift)
    # Set to silence for heading/ tailing
    if shift > 0:
        augmented_chunk[:shift] = 0
    else:
        augmented_chunk[shift:] = 0
    return augmented_chunk


def change_pitch(chunk, sampling_rate, tone_shifts):
    # thanks librosa, for your contribution
    pitch_factor = tone_shifts * 12
    return librosa.effects.pitch_shift(chunk, sampling_rate, pitch_factor)

def change_speed(chunk, speed_factor):
    stretched = librosa.effects.time_stretch(chunk, speed_factor)

    if speed_factor > 1:
        # pad array with zeros to assure correct feature size
        padding = np.zeros([chunk.shape[0] - stretched.shape[0]])
        return np.concatenate([stretched, padding])

    elif speed_factor < 1:
        # trim to meet shape of model input
        stretched = stretched[:chunk.shape[0]]
        return stretched
    else:
        return chunk

def augment_chunks(chunks, sampling_rate, max_time_shift):
    # should be able to multiply our dataset by a very significant factor 
    # inspired by https://medium.com/@makcedward/data-augmentation-for-audio-76912b01fdf6

    start = time.time()
    # separated for future debugging & validation
    chunks_noise = []
    chunks_l_shift = []
    chunks_r_shift = []
    chunks_pitch = []
    chunks_speed = []

    augmented_data = []

    # all operations are done chunk-wise, so must iterate through with functions
    for idx, chunk in enumerate(chunks):
        # injecting noise
        NOISE_FACTOR = 0.1
        chunk_added_noise = inject_noise(chunk, NOISE_FACTOR)
        chunks_noise.append(chunk_added_noise)

        # shifting time left
        shift_max = max_time_shift * 0.25 # in seconds
        chunk_shifted_l = shift_time(chunk, sampling_rate, shift_max, "left")
        chunks_l_shift.append(chunk_shifted_l)

        # shifting time right
        chunk_shifted_r = shift_time(chunk, sampling_rate, shift_max, "right")
        chunks_r_shift.append(chunk_shifted_r)

        # changing pitch
        tone_shifts = np.random.uniform(-1, 1) 
        chunk_changed_pitch = change_pitch(chunk, sampling_rate, tone_shifts)
        chunks_pitch.append(chunk_changed_pitch)

        # changing speed 
        speed_shift = np.random.uniform(0.5, 1.5)
        chunk_speed_shifted = change_speed(chunk, speed_shift)
        chunks_speed.append(chunk_speed_shifted)
    
        # saving data for the correct output
        # path = "data/"
        # wav.write(path + "original_{}.wav".format(idx), sampling_rate, chunk.astype(np.float32)) 
        # wav.write(path + "noise_{}.wav".format(idx), sampling_rate, chunk_shifted_l.astype(np.float32)) 
        # wav.write(path + "shift_l_{}.wav".format(idx), sampling_rate, chunk_shifted_r.astype(np.float32)) 
        # wav.write(path + "shift_r_{}.wav".format(idx), sampling_rate, chunk_changed_pitch.astype(np.float32)) 
        # wav.write(path + "speed_{}.wav".format(idx), sampling_rate, chunk_speed_shifted.astype(np.float32)) 
 


    # combine data for simple return
    augmented_data.extend(chunks)
    augmented_data.extend(chunks_noise)
    augmented_data.extend(chunks_l_shift)
    augmented_data.extend(chunks_r_shift)
    augmented_data.extend(chunks_pitch)
    augmented_data.extend(chunks_speed)
    end = time.time()
    print(f"Runtime of augment_data is: {end - start}")

    return np.array(augmented_data)


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
# may need to bin specific frequency bands that are characteristic for high frequency cyclic items
# this could be greatly beneficial to separate between the two 
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
    return np.array(chunks_Y_binned)

def scale_Y(chunks_grinder_Y: np.array, chunks_env_Y: np.array):
    # scaling may be better to use log scale instead of this -1 to 1 scale but that can be fixed later
    start = time.time()

    sc = StandardScaler()
    # first row is the bin values 
    n_rows_g = chunks_grinder_Y.shape[0]
    n_rows_e = chunks_env_Y.shape[0]

    X_combined = np.vstack([chunks_grinder_Y, chunks_env_Y])
    X_combined_scaled = sc.fit_transform(X_combined)
    
    X_g_scaled = X_combined_scaled[:n_rows_g, :]
    X_e_scaled = X_combined_scaled[(n_rows_g):, :]

    # save scaler for future realtime scaling
    dump(sc, open('audio_scaler_speedy.pkl', 'wb'))
    end = time.time()

    print(f"Runtime of scale_Y is: {end - start}")

    return X_g_scaled, X_e_scaled

def categorize_data(X_grinder, X_env):
    start = time.time()

    class_0 = np.repeat(0, X_env.shape[0])
    class_1 = np.repeat(1, X_grinder.shape[0])

    classes_h = np.hstack([class_0, class_1])
    # reshape to be vertical

    classes = np.vstack(classes_h)
    
    end = time.time()

    print(f"Runtime of categorize_data is: {end - start}")
    return classes

def split_data(X, y):
    # can change this from not being dependent on sklearn if we want to know the training things but 
    # no need to think about that now
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

    return X_train, X_test, y_train, y_test

def build_ann(X):
    nr_features = X.shape[1]
    # Initialization
    ann_clf = tf.keras.models.Sequential()

    # Some arbitrary architecture for now
    ann_clf.add(tf.keras.layers.Input(shape=nr_features))

    ann_clf.add(tf.keras.layers.Dense(units=nr_features/2, activation='relu'))
    ann_clf.add(tf.keras.layers.Dropout(0.2))

    ann_clf.add(tf.keras.layers.Dense(units=nr_features/4, activation='relu'))
    ann_clf.add(tf.keras.layers.Dropout(0.2))

    ann_clf.add(tf.keras.layers.Dense(units=nr_features/8, activation='relu'))
    ann_clf.add(tf.keras.layers.Dropout(0.2))

    # ann_clf.add(tf.keras.layers.Dense(units=nr_features/16, activation='relu'))
    # ann_clf.add(tf.keras.layers.Dropout(0.2))

    # ann_clf.add(tf.keras.layers.Dense(units=30, activation='relu'))
    # ann_clf.add(tf.keras.layers.Dropout(0.2))

    # ann_clf.add(tf.keras.layers.BatchNormalization())
    ann_clf.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    # Compiling
    ann_clf.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    ann_clf.summary()

    return ann_clf

def validation_plots(h):
    # summarize history for accuracy
    plt.plot(h.history['accuracy'])
    plt.plot(h.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def validation_metrics(model, X_t, y_t):
    y_p = (model.predict(X_t) > 0.5)
    CM = confusion_matrix(y_t, y_p)
    print('Confusion Matrix=', CM)

    accuracy = accuracy_score(y_t, y_p)
    print('Accuracy Score=', accuracy)

def save_model_as_tflit(ann_clf):
    ann_clf.save('angle-grinder-detector-2s.h5')
    model = tf.keras.models.load_model('angle-grinder-detector-2s.h5')

    #converting to tflite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    #save tflite model
    tflite_model = converter.convert()
    open('angle-grinder-detector-2s.tflite', 'wb').write(tflite_model)

if __name__ == "__main__":
    #### split audio data into chunks ####
    # conclusion is we need more data
    # Variable data parameters
    SECONDS = 0.5
    BINS = 2000
    #

    total_start = time.time()

    raw_data, raw_data_environment, sample_rate = load_data()
    floor_recordings(raw_data, raw_data_environment, sample_rate)

    chunks_grinder = create_chunks(raw_data, sample_rate, SECONDS)
    chunks_env = create_chunks(raw_data_environment, sample_rate, SECONDS)

    # # # augment data for some reason, augmentation is not helping accuracy or loss
    # interesting response to fft for audio classification: https://dsp.stackexchange.com/questions/8220/performing-classification-based-on-fft-results
    # TODO: investigate what data looks like after being augmented 
    # TODO: switch features to MFCC's instead of frequencies
    # TODO: consider making a multi layered classifier... only analyse .wav if the file has over a defined threshold sound amplitude 
    # chunks_grinder = augment_chunks(chunks_grinder, sample_rate, SECONDS)
    # chunks_env = augment_chunks(chunks_env, sample_rate, SECONDS)

    # Y is a list whose values are individual chunks and columns are frequencies
    chunks_grinder_Y, freqs_grinder  = raw_audio_to_freq(chunks_grinder, sample_rate)
    chunks_env_Y, freqs_env = raw_audio_to_freq(chunks_env, sample_rate)

    # naming convention capital 'X' is concurrent with matrix notation
    X_grinder = bin_chunks_Y(chunks_grinder_Y, BINS)
    X_env = bin_chunks_Y(chunks_env_Y, BINS)

    # scaling X's 
    X_grinder_scaled, X_env_scaled = scale_Y(X_grinder, X_env)

    # create categories for grinder and environment
    y = categorize_data(X_grinder_scaled, X_env_scaled)
    X = np.vstack([X_grinder_scaled, X_env_scaled])
    
    # split data to test and train groups
    X_train, X_test, y_train, y_test = split_data(X, y)

    # build ann model
    ann = build_ann(X)

    history = ann.fit(X_train, y_train, validation_data = (X_test, y_test), \
        batch_size = 100, epochs = 100, \
        verbose = 1)


    # model validation
    validation_plots(history)
    validation_metrics(ann, X_test, y_test)

    total_end = time.time()

    #save_model_as_tflit(ann)

    print(f"Runtime of total program is: {total_end - total_start}")
    print("done")

# %%
