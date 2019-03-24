import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wvf
from scipy.fftpack import dct
from scipy import signal

file_path = "C:\Program Files\Python36\ThirdYearProject\Speech_TIMIT\\train\FCJF0"
file_list = os.listdir(file_path)

for file_name in file_list:
    try:
        file = file_path + "\\" + file_name
        rate, data = wvf.read(file)
        num_samples = len(data)
        framesize = int(0.064*rate)
        overlap = int(0.032 * rate)
        total_frames = int(num_samples/overlap)

        print("Samples of file " + file_name + " is: " + str(num_samples))
        print("Sampling rate of file: " + str(rate))
        print("Time length of file = " + str(len(data) / rate) + "s")
        print("Framesize: " + str(framesize) + " samples per frame")
        print("Overlap: " + str(overlap) + " samples per frame")
        print("Total frames: " + str(total_frames))
        print("\n")

        frames = np.ndarray((total_frames, framesize))
        dft_matrix = np.ndarray((total_frames, framesize))
        abs_dft_matrix = np.ndarray((total_frames, framesize))

        for i in range(total_frames):
            for j in range(framesize):
                if ((i*overlap + j) < num_samples):
                    frames[i][j] = data[i*overlap + j]
                else:
                    frames[i][j] = 0

        saw_filter_a = signal.waveforms.sawtooth(range(len(frames)), width = [0.5])
        saw_filter_b = signal.waveforms.sawtooth(range(len(frames)), width = [0.6])
        for i in range(total_frames):
            dft_matrix[i] = np.fft.fft(frames[i])
            dft_matrix[i] = signal.filtfilt(saw_filter_a, saw_filter_b, dft_matrix[i])
            dft_matrix[i] = dct(dft_matrix[i])
            abs_dft_matrix[i] = abs(dft_matrix[i]) * abs(dft_matrix[i]) / max(abs(dft_matrix[i]))
            abs_dft_matrix[i] = np.log10(abs_dft_matrix[i])


        f = open(file_name[:-4] + ".logFBE", "w+")
        f.writelines(str(abs_dft_matrix))

        t = range(len(abs_dft_matrix))
        plt.plot(t, abs_dft_matrix)
        plt.ylabel("Frequency")
        plt.xlabel("Frame number")

    except Exception as e:
        print("Exception thrown as: " + str(e))
        pass