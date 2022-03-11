# Feb 18 2022

# At first it made sense to not save all the audio data into individual files.
# But now that I'm debugging the models, it's a real pain in the ass to not have
# the files be accessible in a directory. So this script loads the long recordings
# and splits them into equal length .wav files to be processed.

#%%

import os
import time
import numpy as np
import scipy.io.wavfile as wav
import math


def load_data():
    # the 'trimmed' data were manually truncated to the valueable information
    raw_angle_grinders, raw_environ = [], []
    start = time.time()

    for root, dirs, files in os.walk(
        "C:/Users/adamf/OneDrive/Documents/university/UBC/homework_Winter_2021/Term 2/IGEN_330/BikeSentry_data/angle-grinders/"
    ):
        for file in files:
            # samplerate is constant from the same recording device. If not iPhone XR, do not do this!!!!!
            if "trimmed" in file:
                samplerate, y = wav.read(root + file)
                y0 = y[:, 0]
                y1 = y[:, 1]
                raw_angle_grinders.append(y0)
                raw_angle_grinders.append(y1)

            if "envi" in file:
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

        # print(len(data_grinders[index]) / sample_rate)conda

    # repeat for environment
    # cutting data from the front of the recording to make records to be integer seconds
    for index, rec in enumerate(data_environ):
        s = len(rec) / sample_rate
        floor_s = math.floor(s)
        time_cutoff = s - floor_s
        samples_cutoff = int(time_cutoff * sample_rate)
        data_environ[index] = rec[samples_cutoff:]

        # print(len(data_environ[index]) / sample_rate)

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
            chunk = rec[i : (i + chunk_size)]
            chunks_final[idx] = chunk
            idx += 1

    end = time.time()
    print(f"Runtime of create_chunks is: {end - start}")
    return chunks_final


def save_snippets(snippets, samplerate, naming_convention):
    dir = "../data/"
    for idx, snippet in enumerate(snippets):
        f = dir + naming_convention + "_{}.wav".format(idx)
        wav.write(f, samplerate, snippet)


if __name__ == "__main__":
    SECONDS = 0.5

    # load data
    raw_data, raw_data_environment, sample_rate = load_data()

    # make recordings into digestible lengths
    floor_recordings(raw_data, raw_data_environment, sample_rate)

    # make snippets of audio
    chunks_grinder = create_chunks(raw_data, sample_rate, SECONDS)
    chunks_env = create_chunks(raw_data_environment, sample_rate, SECONDS)

    save_snippets(chunks_grinder, sample_rate, "grinder_05")
    save_snippets(chunks_env, sample_rate, "env_05")
