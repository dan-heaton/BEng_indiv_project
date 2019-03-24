import os
from time import time
import numpy as np
from scipy import signal
import scipy.io.wavfile as wvf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import accuracy_score as acc_sco
import warnings
warnings.filterwarnings('ignore')


print("\n\t//\t//\t//\tAritifical Neural Network for Audio Training and Prediction\t//\t//\t//")

num_dimensions = 512

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
            for i in range(total_frames):
                dft_matrix[i] = np.fft.fft(frames[i])
                dft_matrix[i] = signal.filtfilt(saw_filter_a, saw_filter_b, dft_matrix[i])
                temp = abs(dft_matrix[i]) * abs(dft_matrix[i]) / max(abs(dft_matrix[i]))
                temp = np.log10(temp)
                abs_dft_matrix[i] = np.split(temp, 2)[0]
            class_matrices.append(abs_dft_matrix)
        stage_one_matrices.append(class_matrices)
    return stage_one_matrices


def data_preprocessing(data_matrices):
    y_length_zeroes = 0
    y_length_ones = 0
    stage_one_data = []
    for class_num in range(len(data_matrices)):
        for file_num in range(len(data_matrices[class_num])):
            for frame_num in range(len(data_matrices[class_num][file_num])):
                stage_one_data.append(data_matrices[class_num][file_num][frame_num])
                if class_num < (len(data_matrices) / 2):
                    y_length_zeroes += 1
                else:
                    y_length_ones += 1

    x = np.array(stage_one_data)
    zeroes = np.zeros(y_length_zeroes)
    ones = np.ones(y_length_ones)
    y = np.concatenate((zeroes, ones))

    return x,y


#                       ***** STAGE 1 *****

print("\n\nStage 1...\n")
start_time = time()
class_source_train = "C:\Program Files\Python36\ThirdYearProject\Speech_TIMIT\\train\\"
train_classes = os.listdir(class_source_train)
train_matrices = create_data_arrays(class_source_train, train_classes)
x_train, y_train = data_preprocessing(train_matrices)

class_source_test = "C:\Program Files\Python36\ThirdYearProject\Speech_TIMIT\\test\\"
test_classes = os.listdir(class_source_test)
test_matrices = create_data_arrays(class_source_test, test_classes)
x_test, y_test = data_preprocessing(test_matrices)

phase_one_time = time() - start_time
print(str(round(phase_one_time, 2)) + " seconds for stage 1; " +
      str(round(time() - start_time, 2)) + " seconds total")


#                       ***** STAGE 2 - TRAINING THE ANN *****

print("\n\nStage 2...\n")
num_extra_hidden_layers = 3
num_hidden_nodes = 30
ann = Sequential()

ann.add(Dense(units= num_hidden_nodes, activation= 'sigmoid', input_dim= num_dimensions))
for i in range(num_extra_hidden_layers):
    ann.add(Dense(units = num_hidden_nodes, activation= 'sigmoid'))
ann.add(Dense(units= 1, activation= 'sigmoid'))

ann.compile(optimizer= 'adam', loss = 'binary_crossentropy', metrics= ['accuracy'])
ann.fit(x= x_train, y= y_train, batch_size= 10, nb_epoch= 30)

phase_two_time = time() - phase_one_time - start_time
print(str(round(phase_two_time, 2)) + " seconds for stage 2; " +
      str(round(time() - start_time, 2)) + " seconds total")


#                       ***** STAGE 3 - TESTING THE ANN *****

print("\n\nStage 3...\n")

y_predict = ann.predict(x_test)
conf_mat = cm(y_true= y_test, y_pred= y_predict.round())
accuracy = acc_sco(y_true=y_test, y_pred= y_predict.round())

print("\n\nConfusion matrix")
print(conf_mat)
print("\n\nAccuracy")
print(str(round(accuracy*100, 2)) + "%")

phase_three_time = time() - phase_two_time - phase_one_time - start_time
print(str(round(phase_three_time, 2)) + " seconds for stage 3; " +
      str(round(time() - start_time, 2)) +" seconds total")
print("Hidden layers = " + str(num_extra_hidden_layers + 1) + " with " +
      str(num_hidden_nodes) + " nodes each")