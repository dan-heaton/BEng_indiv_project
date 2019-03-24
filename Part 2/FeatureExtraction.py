import os
import numpy as np
import scipy.io.wavfile as wvf
from librosa.feature import chroma_stft
from librosa.feature import chroma_cqt
from python_speech_features import mfcc, fbank, logfbank
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

frame_time_len = 0.02                                       #define the time each row (frame) should equate
                                                            #(note: includes overlap, so true frame time is 0.01s)
reduced_dim = 30                                            #define the number of features to extract from file
frame_len = 882                                             #frame len parameter for 2 functions (882 is default)
audio_processing_choice = "logbank"                         #defines choice of feature extraction functions to use



def create_features():                                      #Function extracts features from all files
    print("\nConverting LM data to features...")            #prints to user when the function starts up
    features_list = []                                      #creates an empty list to hold features from all files
    for i, (file_name) in enumerate(file_names):            #for each smaller .wav file in specified directory
        print("File " + str(i+1) + "...")                   #prints to use what file extracting features from
        full_file_name = source_location + "\\" + \
                         file_name                          #get full name of file, inc directory
        rate, data = wvf.read(full_file_name)               #read in file as .wav as data and its sampling rate

        if audio_processing_choice == "chroma":             #if feature extraction choice is 'chroma',
            data = np.asarray(                              #process data by flattening data to 1D, taking every
                [float(datum) for datum in                  #other value of data, converting each to floats, and
                 data.flatten()[0::2]])                     #converting to a numpy array
            features = chroma_stft(y=data, sr=rate).T       #This modified data is passed to chroma function
                                                            #with sampling rate and result transposed
                                                            #to give ('# frames' x '12 features')
            features = np.repeat(features, 3, axis=1)       #Append several copies of this horizontally to give
                                                            #36 features (enough for CNNv3 to work with)

        elif audio_processing_choice == "cqt":              #else if feature extraction choice is 'cqt',
            data = np.asarray(                              #process data by flattening data to 1D, taking every
                [float(datum) for datum in                  #other value of data, converting each to floats, and
                 data.flatten()[0::2]])                     #converting to a numpy array
            features = chroma_cqt(y=data, sr=rate,          #This modified data is passed to cqt function with
                                  n_chroma=reduced_dim).T   #sampling rate and result transposed to give
                                                            #('# frames' x 'reduced_dim features')

        elif audio_processing_choice == "mfcc":             #else if feature extraction choice is 'mfcc',
            features = mfcc(signal=data, samplerate=rate,   #pass .wav data directly with sampling rate
                            winlen=frame_time_len,          #to 'mfcc' function and result is feature vector as
                            winstep=frame_time_len,         #('# frames' x 'reduced_dim features')
                            numcep=reduced_dim,
                            nfilt=reduced_dim*2,
                            nfft= frame_len)

        elif audio_processing_choice == "fbank":
            features = fbank(signal=data, samplerate=rate,  #else if feature extraction choice is 'fbank',
                             winlen=frame_time_len,         #pass .wav data directly with sampling rate
                             winstep=frame_time_len,        #to 'fbank' function and result is feature vector as
                             nfilt=reduced_dim,             #('# frames' x 'reduced_dim features')
                             nfft=frame_len)[0]             #(with only first item from list as this is the numpy
                                                            #array we're interested in; the other being array
                                                            #of energies in each frame)

        else:                                               #else if feature extraction choice is anything else
            features = logfbank(signal=data,                #pass .wav data directly with sampling rate
                                samplerate=rate,            #to 'logfbank' function and result is feature vector
                                winlen=frame_time_len,      #as ('# frames' x 'reduced_dim features')
                                winstep=frame_time_len,
                                nfilt=reduced_dim,
                                nfft=frame_len)
        features_list.append(features)                      #Add the extracted features of current .wav file to
                                                            #list, and return this list after features
    return features_list                                    #of all files have been extracted



def create_csv(features, target_location):                  #Function takes in features of all .wav files and
                                                            #location in which to place .csvs of feature data
    print("\n\nCreating .csv files...")                     #prints to user when the function starts up
    for i, file in enumerate(features):                     #For each file's feature set
        print("File " + str(i + 1) + "...")                 #print to user what .csv it's currently creating
        label_file = open(label_location +                  #open its corresponding .lab file
                          file_names[i][:-4] +              #and reads its lines to a list, 'label_file'
                          ".lab").readlines()
        label_lines = []                                    #create new empty list of label lines
        for j in range(len(label_file)):                    #for each line in the label file,
            start_num = int(label_file[j].split("\t")       #get the start time of the LM in seconds,
                            [0].strip(' ')) / 10000000
            end_num = int(label_file[j].split("\t")         #get the finish time of the LM in seconds,
                          [1].strip(' ')) / 10000000
            label = label_file[j].split("\t")[2].split      #get the label of the LM,
            ("\n")[0].strip(' ')
            label_lines.append([start_num,                  #and append these as a list into another list
                                end_num, label])

        if (len(label_file) != 0):                          #if the .lab file isn't empty,
            end_time = int(label_file[-1].split("\t")       #get the local end time of the list in seconds
                           [1]) / 10000000
        else:                                               #else, its an 'Audio' .lab file (i.e. no LMs)
            end_time = 40                                   #and thus 40s long
        num_rows = len(file)                                #Count number of rows (frames) in the .wav file,
        time_step = end_time / num_rows                     #calculate time difference between them,
        time_samples = np.arange(0.0, end_time, time_step)  #and create an array of numbers increasing from
                                                            #0.0 to finishing time of .wav with 'time_step'
                                                            #difference between successive numbers

        label_arr = []                                      #create empty list
        for start_time in time_samples:                     #For each time frame in list of time samples,
            start_time = round(start_time, 3)               #round the start time to 3 decimal places,
            finish_time = round(start_time + time_step, 3)  #get finishing time as next number in list,
                                                            #and round this to 3 decimal places
            for line in label_lines:                        #For each line in the label file,
                if start_time >= round(line[0], 1):         #if the time period of the LM file (between start
                    if finish_time <= round(line[1], 1):    #and finish times) falles between start and end
                        label_arr.append(line[2])           #times of a line of corresponding .lab file,
                        break                               #append that line's label to list of labels
                                                            #for frames, and move onto next time period
                                                            #in time_samples
            if len(label_lines) == 0:                       #If there aren't any lines in .lab file,
                label_arr.append("Audio")                   #set corresponding time sample label to 'Audio'

        df1 = pd.DataFrame(file)                            #Create DataFrame object from features of .wav file
        df2 = pd.DataFrame(label_arr)                       #Create DataFrame object from labels for each feature
        csv_file_name = target_location + \
                        (file_names[i])[:-4] + \
                        ".csv"                              #Open a .csv file with same name as .wav but with
                                                            #.csv file extension instead
        df = pd.concat([df1, df2], axis=1)                  #Concat the features and labels horizontally
        df.to_csv(csv_file_name, sep=',',                   #and write this concatenated object to .csv
                  header=False, index=False)                #with no header column or row



source_location = "C:\\Users\\Dan\\Documents\\LMs\\"        #define location of smaller .wav files
label_location = "C:\\Users\\Dan\\Documents\\LMlabels\\"    #define location of corresponding .lab files
file_names = os.listdir(source_location)                    #list of smaller .wav files in directory
target_location = "C:\\Users\\Dan\\Documents\\LMdata\\"     #define location to place new .csvs

features = create_features()                                #create features for every smaller .wav file
for file in os.listdir(target_location):                    #for each file already in location to place .csvs,
    os.remove(target_location + file)                       #remove them from directory
create_csv(features, target_location)                       #and create every .csv from every .wav and .lab file