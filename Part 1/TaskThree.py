import os
import numpy as np
import scipy.io.wavfile as wvf
from math import sqrt, pi, exp
from scipy import signal
from sklearn.mixture import GaussianMixture
from time import time
import warnings
warnings.filterwarnings('ignore')


print("\n\t//\t//\t//\tGaussian Multivariate Mixture Model for Audio Training and Prediction\t//\t//\t//")

num_components = 100
em_iterations = 100
reduced_num_dimensions = 50
tri_size = int(np.floor(512/reduced_num_dimensions) - 1)

def create_data_arrays(class_source, classes):
    stage_one_matrices = []
    for class_name in classes:
        class_matrices = []
        file_list = os.listdir(class_source + "\\" + class_name)
        if not os.path.exists(class_name):
            os.makedirs(class_name)
        for file_name in file_list:
            file = class_source + "\\" + class_name + "\\" + file_name
            rate, data = wvf.read(file)
            num_samples = len(data)
            framesize = int(0.064*rate)
            overlap = int(0.032 * rate)
            total_frames = int(num_samples/overlap)

            frames = np.ndarray((total_frames, framesize))
            dft_matrix = np.ndarray((total_frames, framesize))
            abs_dft_matrix = np.ndarray((total_frames, int(framesize/2)))

            for i in range(total_frames):
                for j in range(framesize):
                    if ((i*overlap + j) < num_samples):
                        frames[i][j] = data[i*overlap + j]
                    else:
                        frames[i][j] = 0

            saw_filter_a = signal.waveforms.sawtooth(range(len(frames)), width = [0.5])
            saw_filter_b = signal.waveforms.sawtooth(range(len(frames)), width = [0.5001])
            tri = signal.triang(M=tri_size)
            file_matrices = []
            for i in range(total_frames):
                dft_matrix[i] = np.fft.fft(frames[i])
                dft_matrix[i] = signal.filtfilt(saw_filter_a, saw_filter_b, dft_matrix[i])
                temp = abs(dft_matrix[i]) * abs(dft_matrix[i]) / max(abs(dft_matrix[i]))
                temp = np.log10(temp)
                abs_dft_matrix[i] = np.split(temp, 2)[0]
                final_dft_matrix = []
                for j in range(reduced_num_dimensions):
                    final_dft_matrix.append(np.dot(abs_dft_matrix[i][j*tri_size:(j+1)*tri_size], tri.T))
                file_matrices.append(final_dft_matrix)
            class_matrices.append(file_matrices)
        stage_one_matrices.append(class_matrices)
    return stage_one_matrices


#                       ***** STAGE 1 - FEATURE EXTRACTION FOR TRAINING DATA*****

print("\n\nStage 1...\n")

start_time = time()
class_source_train = "C:\Program Files\Python36\ThirdYearProject\Speech_TIMIT\\train\\"
train_classes = os.listdir(class_source_train)
stage_one_matrices = create_data_arrays(class_source_train, train_classes)
phase_one_time = time() - start_time
print(str(round(phase_one_time, 2)) + " seconds for stage 1; " + str(round(time() - start_time, 2)) + " seconds total")


#                       ***** STAGE 2 - TRAINING THE GMMS *****

print("\n\nStage 2...\n")

weights, means, covariances = [], [], []

for i in range(2):
    temp_matrix = []
    for j in range(2):
        class_matrices = stage_one_matrices[2*i + j]
        for file_matrix in class_matrices:
            for frame in file_matrix:
                temp_matrix.append(frame)
    gmm = GaussianMixture(n_components = num_components, max_iter = em_iterations, covariance_type = 'full')
    gmm.fit(np.asarray(temp_matrix))
    print("GMM " + str(i+1) + " fitted")

    weights.append(gmm.weights_)
    means.append(gmm.means_)
    covariances.append(gmm.covariances_)

phase_two_time = time() - phase_one_time - start_time
print(str(round(phase_two_time, 2)) + " seconds for stage 2; " +
      str(round(time() - start_time, 2)) + " seconds total")


#                       ***** STAGE 3 - FEATURE EXTRACTION AND TESTING OF GMMS*****

print("\n\nStage 3...\n")

class_source_test = "C:\Program Files\Python36\ThirdYearProject\Speech_TIMIT\\test\\"
test_classes = os.listdir(class_source_test)
stage_three_matrices = create_data_arrays(class_source_test, test_classes)
conf_matrix = np.zeros((2,2))
equals_prob = 0


for class_num in range(len(stage_three_matrices)):                #for each class in 'test'
    print("Class " + str(class_num+1) + "...")
    for file_num in range(len(stage_three_matrices[class_num])):  #for each file within that class
        print("\tFile " + str(file_num+1) + "...")
        frame_matrix = stage_three_matrices[class_num][file_num]
        for frame_num in range(len(frame_matrix)):                #for each frame within the file

            frame = frame_matrix[frame_num]
            p_y_c_arr = []
            for i in range(2):                                    #for each GMM to compare it to

                p_y_c = 0
                for j in range(num_components):                   #for each component of the GMM
                    expon = 0
                    sigma = 1
                    for dim_num in range(reduced_num_dimensions):         #for each dimension of the frame
                        yt_n = frame[dim_num]           #FIX MEEEEEEEEE
                        mean = means[i][j][dim_num]
                        var = covariances[i][j][dim_num][dim_num]
                        expon += ((yt_n - mean) ** 2) / var
                        sigma *= abs(var)
                    sqrt_abs_sigma = sqrt(abs(sigma))
                    denominator = ((2 * pi) ** (reduced_num_dimensions/2)) * sqrt_abs_sigma
                    p_yt_c = np.float64((1 / denominator) * exp(-0.5 * expon))
                    p_y_c += np.float64(p_yt_c * weights[i][j])
                p_y_c_arr.append(p_y_c)
            p_c1, p_c2 = np.float64(0.5), np.float64(0.5)
            p_y = np.float64(p_c1*p_y_c_arr[0] + p_c2*p_y_c_arr[1])

            p_c1_y = np.float64(p_c1*p_y_c_arr[0] / p_y)
            p_c2_y = np.float64(p_c2*p_y_c_arr[1] / p_y)

            if p_c1_y > p_c2_y:
                conf_matrix[class_num][0] += 1
            elif p_c1_y < p_c2_y:
                conf_matrix[class_num][1] += 1
            else:
                equals_prob += 1

correct_nums = conf_matrix[0][0] + conf_matrix[1][1]
total_nums = conf_matrix[0][0] + conf_matrix[0][1] + conf_matrix[1][0] + conf_matrix[1][1] + equals_prob
accuracy = correct_nums / total_nums

print("\n")
print("Final confusion matrix")
print(conf_matrix[0])
print(conf_matrix[1])
print("Accuracy of: " + str(correct_nums) + " out of " + str(total_nums) +
      " = " + str(round((accuracy * 100), 2)) + "% accuracy\n")

phase_three_time = time() - phase_two_time - phase_one_time - start_time
print(str(round(phase_three_time, 2)) + " seconds for stage 3; " +
      str(round(time() - start_time, 2)) +" seconds total")