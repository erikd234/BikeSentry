# The ethos of this training regime is to take what I know about the data
# and create a biased way of preprocessing in order to make loud, cyclic
# audio very recognizable. 

# I need to look at recordings of other cyclic things such as saws, drills,
# mills, etc. to see if they differ a lot between each other

# I also need to look at normal environmental sounds as well as common
# noise from our microphone. 

# Fine scale features may not be necessary at all so doing a translation subtraction
# of amplitude from an FFT might be beneficial to get rid of noise

# Feature ideas:
# - count of frequencies that have amplitudes higher than a threshold
# - bound frequencies that we care about (e.g. get rid of 0-1kHz which is common in human speech)
# 

# The average maximum amplitude in environment noise: 0.003119
# '      '       '         '     ' angle grinder    : 0.0176
# They're on a different order of magnitude 
# try: y_grinder - mean_max_y_env & y_env - mean_max_y_env

# Feb 13 2022
# Written by Adam Fong
# Built off of binary_classifier_speedy.py

# Well, tensorflow is harder to get running on a raspberry pi than I thought it would be
# Alternate route that has more interesting features is classifying the MFCC's of audio
#%%
import scipy.io.wavfile as wav
from scipy import stats
import numpy as np
import os
import math
from matplotlib import pyplot as plt
from joblib import dump
import pickle
import time 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import librosa 
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier


def load_data():
    # the 'trimmed' data were manually truncated to the valueable information
    raw_angle_grinders, raw_environ = [], []
    start = time.time()

    for root, dirs, files in os.walk("../data/"):
        for file in files:  
            # samplerate is constant from the same recording device. If not iPhone XR, do not do this!!!!!
            if("grinder" in file):
                samplerate, y = wav.read(root + file)
                raw_angle_grinders.append(y)
        
            if("env" in file):
                samplerate, y = wav.read(root + file)
                raw_environ.append(y)
    
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
    return np.array(chunks_Y), np.array(chunks_freqs)

def remove_freqs(amplitudes, freqs, min, max):
    '''
    Get rid of frequencies between min & max 
    '''
    f = ~((freqs >= min) & (freqs <= max))[0]
    return amplitudes[:,f], freqs[0, f]

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
    #pickle.dump(sc, open('mfcc_scaler.pkl', 'wb'))
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

def build_svc(X, y):
    
    cls = SVC(kernel = "rbf").fit(X, y)

    return cls

def build_rf(X, y):
    cls = RandomForestClassifier()
    cls.fit(X, y)
    
    return cls

def build_knn(X, y):
    cls = KNeighborsClassifier()
    cls.fit(X, y)
    
    return cls

def build_sgd(X, y):
    cls = SGDClassifier(random_state = 0)
    cls.fit(X, y)
    
    return cls

def n_max(arr, n):
    """ Find the n largest maximum value in an array. Intented to be used with np.apply_along_axis()

    Args:
        arr (np.array): 1 dimensional np array
        n (np.array): n = 1 gets the max value, n = 2 gets the second max value, etc
        
    Returns:
        nth max value in the array
    """
    s = np.sort(arr)
    m = s[-n]
    
    return m

def predictor_1(amplitudes):
    """Find the maximum column in each row of np.array
    Args:
        amplitudes (np.array): 2d array with cols as amplitudes and rows as samples

    Returns:
        np.array: maximum rowwise amplitudes from input array
    """
    p1 = np.apply_along_axis(n_max, 1, amplitudes, 1)
    
    return p1

def predictor_2(amplitudes):
    p2 = np.apply_along_axis(n_max, 1, amplitudes, 2)
    
    return p2
    
def predictor_3(amplitudes):
    p3 = np.apply_along_axis(n_max, 1, amplitudes, 3)
    
    return p3

def get_predictors(amplitudes):
    n_preds = 3
    # create empty array to populate
    X = np.zeros((amplitudes.shape[0], n_preds))
    
    X[:, 0] = predictor_1(amplitudes)
    X[:, 1] = predictor_2(amplitudes)
    X[:, 2] = predictor_3(amplitudes)

    return X

if __name__ == "__main__":
    #### split audio data into chunks ####
    # conclusion is we need more data
    # Variable data parameters
    SECONDS = 0.5

    total_start = time.time()

    snippets_grinder, snippets_env, sample_rate = load_data()

    X_grinder, freqs_grinder = raw_audio_to_freq(snippets_grinder, sample_rate)
    X_env, freqs_env = raw_audio_to_freq(snippets_env, sample_rate)
    
    # remove human speech frequencies 0 - 1000 kHz
    min_freq = 0 #kHz
    max_freq = 1000 #kHz
    
    X_grinder_trunc, freqs_grinder_trunc = remove_freqs(X_grinder, freqs_grinder, min_freq, max_freq)
    X_env_trunc, freqs_env_trunc = remove_freqs(X_env, freqs_env, min_freq, max_freq)
    
    # predictor creation
    # Predictor 1: max amplitude 
    # Predictor 2: second max amplitude
    # Predictor 3: third max amplitude
    
    X_g = get_predictors(X_grinder_trunc)
    X_e = get_predictors(X_env_trunc)
    
    # convert to dB scale
    X_g = 20 * np.log(X_g)
    X_e = 20 * np.log(X_e)
    
    # scaling X's 
    X_g_scaled, X_e_scaled = scale_Y(X_g, X_e)

    # create categories for grinder and eironment
    y = categorize_data(X_g_scaled, X_e_scaled)
    X = np.vstack([X_g_scaled, X_e_scaled])
    
    # split data to test and train groups
    X_train, X_test, y_train, y_test = split_data(X, y)

    # build classifier
    svc = build_svc(X_train, y_train.ravel())
    print("SVC Accuracy: {}".format(svc.score(X_test, y_test)))

    # testing many modelsf
    knn = build_knn(X_train, y_train.ravel())
    print("KNN Accuracy: {}".format(knn.score(X_test, y_test)))

    rf = build_rf(X_train, y_train.ravel())
    print("Random Forest Accuracy: {}".format(rf.score(X_test, y_test)))
    
    sgd = build_sgd(X_train, y_train.ravel())
    print("SGD Accuracy: {}".format(sgd.score(X_test, y_test)))
    
    # confusion matrix
    print("SVC Confusion Matrix")
    print(confusion_matrix(y, svc.predict(X)))
    
    # confusion matrix
    print("KNN Confusion Matrix")
    print(confusion_matrix(y, knn.predict(X)))
    
    # confusion matrix
    print("Random Forest Confusion Matrix")
    print(confusion_matrix(y, rf.predict(X)))
    
    # confusion matrix
    print("SGD Confusion Matrix")
    print(confusion_matrix(y, sgd.predict(X)))
    
    total_end = time.time()

    # save model
    #dump(svc, 'binary_classifier.joblib')
    

    print(f"Runtime of total program is: {total_end - total_start}")
    print("done")


# %%
