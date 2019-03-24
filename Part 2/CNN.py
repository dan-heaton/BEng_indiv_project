import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import mode
import numpy as np
from collections import Counter
import random
from time import time

#Hyperparameters - preprocessing
window_size = 40
window_step = 2
lms_to_classify = ["Nibelungen", "Mime", "Jugendkraft",
                   "Grubel", "Sword"]
num_label_types = len(lms_to_classify) + 1
mode_or_mid = "mid"
one_hot_or_label = "label"
train_test_ratio = 0.2

#Hyperparameters - convolution/pooling layers
num_conv_pool_layers = 4
num_filters = [5, 25, 125, 625]
conv_size = [[3,3], [3,3], [3,3], [3,3]]
conv_stride = 1
conv_padding = "same"
conv_activation = tf.nn.relu
pool_size = [2,2]
pool_stride = 2
pool_padding = "valid"

#Hyperparameters - FC layers
num_fc_layers = 2
size_fc_layers = [400, 400]
fc_activation = tf.nn.relu
dropout_rate = 0.1
output_activation = None

#Hyperparameters - network training
optimizer_learning_rate = 0.001
batch_size = 128
epochs = 80
num_steps = 16000
device_used = "GPU"

data_file_location = "C:\\Users\\Dan\\Documents\\LMdata\\"  #Sets location of .csv data files
tf.logging.set_verbosity(tf.logging.INFO)                   #Sets to give progress output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'                    #Filters out info and warning logs

if device_used == "CPU":                                    #Selects device to use between CPU and GPU
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"



def preprocess_data(data_file_location):                    #Function takes in location of .csvs containing
                                                            #feature data from corresponding .wav files
    X_concat, Y_concat = [], []                             #X and Y lists for ALL data
    frame_times = []                                        #Create empty frame times list
    for file_name in os.listdir(data_file_location):        #For each data file
        data = pd.read_csv(data_file_location + file_name,  #read in the .csv with no header or index
                           header=None, index_col=None)
        for i in range(0, len(data)-window_size,            #from 0 to (end of file-window size) by 'window_step'
                       window_step):
            X = data.iloc[i:i+window_size, :-1].values      #read in the X and Y values as two lists
            Y = data.iloc[i:i+window_size, -1:].values
            frame_times.append(get_frame_time               #append the start time of the current 2D
                               (file_name, i))              #'X' sample system currently working with
            for a in range(len(Y)):                         #for every position in the Y values
                for b in range(len(Y[a])):                  #for the one item in 'a'
                    if Y[a][b] not in lms_to_classify:      #if not in 'lms_to_classify', reassign as "Audio"
                        Y[a][b] = "Audio"
            if mode_or_mid == "mode":                       #If using mode for determining the Y list's label
                Y = mode(Y)[0][0][0]                        #reassign Y values list to a single Y mode value
            else:
                Y = Y[int((len(Y)-1)/2)][0]                 #else assign to Y to middle value in Y values list
            X_concat.append(X)                              #Add X values list and Y value to overall data lists
            Y_concat.append(Y)
    X_concat = np.array(X_concat)                           #Reshape lists into numpy arrays and flatten Y one
    Y_concat = np.array(Y_concat).flatten()
    X_concat, Y_concat, fts = data_undersampling(           #Undersamples X and Y data to remove excess
        X_concat, Y_concat, frame_times)                    #excess 'Audio' frames
    le_map = {}                                             #Create empty label encoder dictionary
    if one_hot_or_label == "label":                         #If choice is to use label encoder,
        le = LabelEncoder()                                 #create label encoder object,
        le.fit(Y_concat)                                    #fit it to the Y data,
        le_map = dict(zip(le.classes_,                      #and add mapping of LM labels (keys)
                          le.transform(le.classes_)))       #and number encodings (values)
        Y_concat = le.transform(Y_concat)                   #Transform Y data into encoded values
    elif one_hot_or_label == "one_hot":                     #Else if choice is one hot,
        Y_concat = OneHotEncoder().\
            fit_transform(Y_concat)                         #create OneHotEncoder object, fit to Y
                                                            #and encode Y to one hot values
    X_train, X_test, Y_train, Y_test = \
        train_test_split(X_concat, Y_concat,                #Split the X and Y 'concat' lists into train
            test_size=train_test_ratio, random_state=45)    #and split data based on chosen
                                                            #ratio and return them
    return X_train, X_test, Y_train, Y_test, X_concat, Y_concat, fts, le_map



def get_frame_time(file_name, i):
    file_start_time = int(file_name.split("_")[1].split     #Gets the global start time of .wav file in ms
                          ("to")[0])*1000
    time_offset = (window_size / 2)                         #Computes time offset from half of window size
    frame_time = file_start_time + (i*window_step) + \
                 time_offset                                #Computes exact time of frame in ms
                                                            #with respect to the source .wav
    return frame_time                                       #and returns this



def data_undersampling(X, Y, fts):                          #Function takes in X and Y lists of feature data
                                                            #and the start time of the 2D X data
    lab_freq = Counter(Y).most_common()                     #Creates dictionary with LM labels and frequencies
    print("Labels and freq in Y: ", lab_freq)               #in Y in descending order and prints this to user
    Z = list(zip(X,Y, fts))                                 #Zips together 2D frames with single LM label
    random.shuffle(Z)                                       #and frame time, converts to list, and shuffles it

    X,Y,fts = zip(*Z)                                       #The shuffled 'Z' is then returned to prev state
                                                            #with X/Y lists + frame time now shuffled same way
    tar_x_len = len(X) - (lab_freq[0][1] - lab_freq[1][1])  # his calculates the amount of samples to have in
                                                            #new 'X' array after excess 'Audio' samples removed
    new_x, new_y, new_fts = [], [], []                      #Creates empty lists for new reduced X and Y lists
                                                            #and new frame times
    num_most_common_added = 0                               #Initializes variable at 0 for number of most
    i = 0                                                   #common LM added, along with an interator variable

    while(len(new_x) < tar_x_len):                          #While the new array still to add more frames,
        if Y[i] != lab_freq[0][0]:                          #if the frame is not an 'Audio' frame,
            new_x.append(X[i])                              #append X sample and Y sample to new X and Y arrays
            new_y.append(Y[i])                              #and the sample time to new time samples array
            new_fts.append(fts[i])
        else:                                               #else if it is an 'Audio' frame,
            if num_most_common_added < lab_freq[1][1]:      #and if havent already added more than most common LM
                new_x.append(X[i])                          #append X sample and Y sample to new X and Y arrays
                new_y.append(Y[i])                          #and the sample time to new time samples array
                new_fts.append(fts[i])
                num_most_common_added += 1                  #increment the number of 'Audio' samples included
        i += 1                                              #increment iterator variable
    new_x = np.asarray(new_x)                               #Convert both X and Y lists to numpy arrays
    new_y = np.asarray(new_y)
    new_lab_freq = Counter(new_y).most_common()             #Re-prints the dictionary with LM labels and
    print("New labels and freq in Y (after minimisation): ",#frequencies in Y in descending order (after
          new_lab_freq)                                     #excess 'Audio' samples are removed from X and Y)
    return new_x, new_y, new_fts                            #Returns the new X/Y arrays and frame times



def build_cnn(features, labels, mode):
    if mode == tf.estimator.ModeKeys.TRAIN:                 #Set variable to enable dropout layers
        is_training = True                                  #to activate if CNN is training
    else:
        is_training = False

    layers = []                                             #Creates empty list to hold each layer of network
                                                            #so layers can be linked together via 'for' loops
    x = features["x"]                                       #Get the 'x' data the CNN will work with
    x_shape = x.get_shape().as_list()                       #Identify its matrix shape as a list
    x_4d = tf.reshape(x, shape=[-1, x_shape[1],
                                x_shape[2], 1])             #Reshape from shape (a,b,c) to (a,b,c,1)
    x_4d = tf.cast(x_4d, tf.float32)                        #Change values in 'x' from ints to floats
    layers.append(x_4d)                                     #Adds 'x' data to layers

    for i in range(num_conv_pool_layers):                   #For however many conv/pool layers are set,
        conv = tf.layers.conv2d(layers[-1],                 #Creates ith convolution layer with kernel,
                                kernel_size=conv_size[i],   #feature maps produced, padding, stride len,
                                filters=num_filters[i],     #and activation function as set by parameters,
                                strides=conv_stride,        #with each layer linked to last layer in 'layers'
                                padding=conv_padding,
                                activation=conv_activation)
        layers.append(conv)                                 #Appends this layer to end of 'layers'
        pool = tf.layers.max_pooling2d(layers[-1],          #Creates ith pooling layer with window size,
                                       pool_size=pool_size, #strides, and padding as set by parameters
                                       strides=pool_stride,
                                       padding=pool_padding)
        layers.append(pool)                                 #Appends this layer to end of 'layers'
    flat = tf.layers.flatten(layers[-1])                    #Flatten pooling output onto a 1D tensor
    layers.append(flat)                                     #Appends this layer to end of 'layers'

    for i in range(num_fc_layers):                          #For however many FC layers are set,
        drop = tf.layers.dropout(layers[-1],                #Creates ith dropout layer with dropout
                                 rate=dropout_rate,         #rate and when to activate as set by parameters
                                 training=is_training)
        layers.append(drop)                                 #Appends this layer to end of 'layers'
        fc = tf.layers.dense(layers[-1],                    #Creates ith FC layer with # of nodes and
                             size_fc_layers[i],             #activation function as set by parameters
                             activation=fc_activation)
        layers.append(fc)                                   #Appends this layer to end of 'layers'
    logits = tf.layers.dense(layers[-1],                    #Creates output layer with # nodes equal to
                             units=num_label_types,         #number of classes to classify and with
                             activation=output_activation)  #activation function as set by parameters

    pred_probs = tf.nn.softmax(logits, name="y_pred")       #Gets output probabilities for classes
    pred_cls = tf.argmax(pred_probs, axis=1)                #Gets most probable class from probs
    if mode == tf.estimator.ModeKeys.PREDICT:               #If predicting, return here with
        predictions = {'pred_probs': pred_probs,            #predicted classes and probabilities
                       'pred_cls': pred_cls}
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions=predictions)

    loss_op = tf.losses.sparse_softmax_cross_entropy(       #Computed loss via loss function based
        labels=labels, logits=logits)                       #on labels and output layer
    optimizer = tf.train.AdamOptimizer(                     #Define optimizer for training
        learning_rate=optimizer_learning_rate)
    train_op = optimizer.minimize(                          #Train CNN via using optimizer to
        loss_op, global_step=tf.train.get_global_step())    #minimize the loss
    acc_op = tf.metrics.accuracy(                           #Get accuracy from labels and pred classes
        labels=labels, predictions=pred_cls)
    fns = tf.metrics.false_negatives(                       #Get false negatives, false positives,
        labels=labels, predictions=pred_cls)                #true negatives, true positives,
    fps = tf.metrics.false_positives(                       #precision, and recall from label types
        labels=labels, predictions=pred_cls)                #and predicted class for each input sample
    tns = tf.metrics.true_negatives(
        labels=labels, predictions=pred_cls)
    tps = tf.metrics.true_positives(
        labels=labels, predictions=pred_cls)
    prec = tf.metrics.precision(
        labels=labels, predictions=pred_cls)
    reca = tf.metrics.recall(
        labels=labels, predictions=pred_cls)
    estim_specs = tf.estimator.EstimatorSpec(               #Return predicted classes, loss funct,
        mode=mode, predictions=pred_cls, loss=loss_op,      #optimizer, and accuracy (plus other metrics)
        train_op=train_op,                                  #of the CNN as an EstimatorSpec object
        eval_metric_ops={'accuracy': acc_op, 'fns': fns,
                         'fps': fps, 'tns': tns,
                         'tps': tps, 'prec': prec,
                         'reca': reca})
    return estim_specs



def predict_lms(predictions, expected, fts, le_map):
    preds, expects = [], []                                 #Create empty predicts and expects lists
    for pred, expect in zip(predictions, expected):         #For each prediction and expected value,
        preds.append(pred['pred_cls'])                      #append predicted and expected values
        expects.append(expect)                              #to respective lists
    fts, preds, expects = (                                 #Sort each list by ascending frame start
        list(t) for t in zip(                               #time in the same manner
        *sorted(zip(fts, preds, expects))))

    sts, ets, lbls = [], [], []                             #Create empty start/end time and label lists
    lm_predicts, lm_trues = {}, {}                          #Creates empty LM predictions and expected
                                                            #value dictionaries
    window = [0,0]                                          #Initialize window as a list with 2 numbers
    i = 0                                                   #Initialize iterator variable

    while window[1] < (len(preds)-1):                       #While the window hasn't reached end of preds
        if preds[i] == preds[i+1]:                          #if there are two consecutive predictions,
            window[1] += 1                                  #extend the window and
            i += 1                                          #increment the iterator variable
        else:                                               #otherwise, if next prediction is different
            if preds[i] != 0:                               #and is not an 'Audio' prediction,
                sts.append(                                 #append its start time to start times list
                    mill_to_str_time(fts[window[0]]))       #as a string of "hh:mm:ss:msmsms"
                ets.append(                                 #append its end time to end times list
                    mill_to_str_time(fts[window[1]]+2))     #as a string of "hh:mm:ss:msmsms"
                for lm_name in le_map:                      #For each LM label in mapping,
                    if preds[i] == le_map[lm_name]:         #if the prediction equals the label encoding
                        lbls.append(lm_name)                #append its corresponding label to labels list
                        if lm_name in lm_predicts:          #If the decoded LM label is in dictionary,
                            lm_predicts[lm_name] += 1       #increase its corresponding value (count)
                        else:                               #else, add it to dictionary with count of 1
                            lm_predicts[lm_name] = 1
            window[0] = window[1]+1                         #Set start of window to previous end of window
            window[1] = window[0]                           #and end of window to same value as start
            i = window[0]                                   #Set iterator to start of window
    df = pd.concat([pd.DataFrame(sts), pd.DataFrame(ets),   #Concatenate lists as DataFrame
                    pd.DataFrame(lbls)], axis=1)            #objects horizontally as DataFrame object
    df.to_csv("C:\\Users\\Dan\\Dropbox\\Uni stuff\\"
              "EE3P\\LM Detections.csv", sep=',',           #Write this DataFrame object as a .csv file
              header=["LM Start Time", "LM End Time",       #as 'LM Detections.csv' with headers
                      "Predicted LM"], index=False)

    new_fts, new_preds, new_expects, new_equals = \
        [], [], [], []                                      #Create empty frame times, predictions, expected
                                                            #values, and equals lists
    for ft in fts:                                          #For each frame time in list,
        new_fts.append(mill_to_str_time(ft))                #append this as a string of "hh:mm:ss:msmsms"
                                                            #to new list of frame times
    for pred, expect in zip(preds, expects):                #For each prediction and expected value,
        for lm_name in le_map:                              #for each LM label in mapping,
            if pred == le_map[lm_name]:                     #if the prediction equals the LM name
                new_preds.append(lm_name)                   #append the LM label to prediction list
            if expect == le_map[lm_name]:                   #if the expected value equals the LM name
                new_expects.append(lm_name)                 #append the LM label to expected value list
        if pred == expect:                                  #If the predicted label for frame equals
            new_equals.append("True")                       #the expected label for frame, append 'True'
        else:                                               #to list, else append 'False'
            new_equals.append("False")
    df = pd.concat([pd.DataFrame(new_fts),                  #Concatenate the frame start times,
                    pd.DataFrame(new_preds),                #predicted LM labels, expected LM labels,
                    pd.DataFrame(new_expects),              #and whether each are the same ('True'/'False')
                    pd.DataFrame(new_equals)], axis=1)      #horizontally as DataFrame objects as
                                                            #one DataFrame object
    df.to_csv("C:\\Users\\Dan\\Dropbox\\Uni stuff\\"        #Write this DataFrame object as a .csv file
              "EE3P\\LM Frame Predictions.csv",sep=',',     #as 'LM Frame Predictions.csv' with headers
              header=["Times (0.2s LM duration)",
                      "Predicted LM", "Expected LM",
                      "Predict == Expected?"], index=False)

    corrects, incorrects = 0, 0                             #Initializes corrects and incorrects variabls
    for i, pred_cls in enumerate(preds):                    #For each predicted frame,
        true_cls = expects[i]                               #get the true frame at the same frame time
        if pred_cls == true_cls:                            #If they are the same, increment corrects,
            corrects += 1                                   #else increment incorrects
        else:
            incorrects += 1
    accur = round((corrects / (corrects + incorrects))      #Calculated the percentage of total frames
                  * 100, 2)                                 #are correctly predicted, rounded to
                                                            #2 decimal places
    print("\nPredicted LMs in source =", lm_predicts)       #Print to the user to dictionary of complete
    print("Prediction accuracy over whole file = " +        #LMs predictions, along with the accuracy of
          str(accur) + "%")                                 #frame predictions on the whole file



def mill_to_str_time(ft):                                   #Function takes in a frame time as ms
    hrs = max(int(ft / 3600000), 0)                         #Extracts hours in frame time,
    ft -= hrs * 3600000                                     #subtracts hours from frame time,
    mins = max(int(ft / 60000), 0)                          #extracts minutes in frame time,
    ft -= mins * 60000                                      #subtracts minutes from frame time
    secs = max(int(ft / 1000), 0)                           #extracts seconds in frame time,
    ft -= secs * 1000                                       #subtracts seconds from frame time,
    return (str(hrs) + ":" + str(mins) + ":"                #and appends these together with
            + str(secs) + ":" + str(int(ft)))               #remainder of frame time in ms
                                                            #as string with ":" between numbers
                                                            #and returns the string



def main():
    print("\n\\\t\\\t\\\tConvolution Neural Network for "   #Prints message on startup of the name
          "Leitmotiv Detection - Version 3\t\\\t\\\t\\\n")  #of the model

    print("\nStage 1: Extracting data from .csv's...\n")    #Prints message to user when Stage 1 starts
    total_time = 0                                          #Initializes variable for total time taken
    start_time = time()                                     #and starts the timer object
    X_train, X_test, Y_train, Y_test, \
    X, Y, fts, le_map = preprocess_data(data_file_location) #Preprocess data from all files into
                                                            #X and Y, training and testing partitions

    labels = {}                                             #Creates an empty dictionary
    for lbl in Y_test:                                      #For each label in 'Y_test',
        if lbl in labels:                                   #if it exists in the dictionary with frequency value,
            labels[lbl] += 1                                #add to this value
        else:                                               #else, add to dictionary with frequency = 1
            labels[lbl] = 1
    print("Labels and occurences in Y_test", labels)        #Prints to user the LM labels with their respective
    print("Total labels =", len(Y_test))                    #frequencies, along with total number of labels

    train_input = tf.estimator.inputs.numpy_input_fn(       #Define the training input to CNN
        x={"x": X_train}, y=Y_train,                        #from output of 'preprocess_data'
        batch_size=batch_size, num_epochs=epochs,           #with set batch size and # train epochs
        shuffle=True)
    test_input = tf.estimator.inputs.numpy_input_fn(        #Define the testing input to CNN
        x={"x": X_test}, y=Y_test,                          #in same way as above
        batch_size=batch_size, num_epochs=epochs,
        shuffle=True)

    cnn = tf.estimator.Estimator(build_cnn)                 #Setup the CNN object based on 'build_cnn' function
    stage_one_time = time() - start_time                    #Calculate time taken for Stage 1 to complete
    total_time += stage_one_time                            #and add to cumulative total time taken
    print("\nTime for stage 1:", round(stage_one_time, 2),  #Print both calculated above values to user
          "seconds,", round(total_time, 2), "total time\n")


    print("\nStage 2: Training CNNv2...\n")                 #Prints message to user when Stage 2 starts
    cnn.train(input_fn=train_input, steps=num_steps)        #Train CNN on training input data with set # steps
    stage_two_time = time() - stage_one_time - start_time   #Calculate time taken for Stage 2 to complete
    total_time += stage_two_time                            #and add to cumulative total time taken
    print("\nTime for stage 2:", round(stage_two_time, 2),  #Print both calculated above values to user
          "seconds", round(total_time, 2), "total time")


    print("\n\nStage 3: Determining accuracy...\n")         #Prints message to user when Stage 3 starts
    accuracy = cnn.evaluate(input_fn=test_input)            #Evaluates for accuracy on testing input
    print("\n\nAccuracy =",
          round(accuracy['accuracy'] * 100, 2), "%")
    print("Loss =", accuracy['loss'])                       #Prints accuracy and loss to user

    total = accuracy['fns'] + accuracy['fps'] + \
            accuracy['tns'] + accuracy['tps']               #Calculates total number of predictions made
    correct = accuracy['tns'] + accuracy['tps']             #and number of those deemed 'correct'
    print("Proportion of correctly classified as an "
          "allowed LM: " + str(correct) + "/"               #Prints to user amount of frames classified
          + str(total) + " = " + str(correct / total))      #correctly as being a LM (but not specific type)

    print("\nTrue positives =", accuracy['tps'])            #Prints to user the number of true positives, true
    print("True negatives =", accuracy['tns'])              #negatives, false positives, and false negatives
    print("False positives =", accuracy['fps'])
    print("False negatives =", accuracy['fns'], "\n")

    f_measure = 2 * ((accuracy['prec'] * accuracy['reca'])  #Calculates the F-measure from above values
                     /(accuracy['prec'] + accuracy['reca']))
    print("Precision (true pos / all pos guesses) =",       #and prints this (rounded to 3 decimal places,
          accuracy['prec'])                                 #along with the precision and recall to the user
    print("Recall (true pos / all true pos) =",
          accuracy['reca'])
    print("F-measure =", round(f_measure, 3))

    stage_three_time = time() - stage_two_time - \
                       stage_one_time - start_time          #Calculate time taken for Stage 3 to complete
    total_time += stage_three_time                          #and add to cumulative total time taken
    print("\nTime for stage 3:",                            #Print both calculated above values to user
          round(stage_three_time, 2),
          "seconds", round(total_time, 2), "total time")


    print("\n\nStage 4: Making predictions...\n")           #Prints message to user when Stage 4 starts
    total_input = tf.estimator.inputs.numpy_input_fn(       #Define the total input to CNN
        x={"x": X}, y=Y, batch_size=batch_size,             #from output of 'preprocess_data'
        num_epochs=1, shuffle=False)                        #with set batch size and # train epochs

    predictions = cnn.predict(input_fn=total_input)         #Get predicted classifications for each frame
                                                            #from the total data input (train and test)
    predict_lms(predictions, Y, fts, le_map)                #Create prediction files and console output
                                                            #from predictions, true values, frame times,
                                                            #and the label encoder map
    stage_four_time = time() - stage_three_time - \
                      stage_two_time - stage_one_time - \
                      start_time                            #Calculate time taken for Stage 4 to complete
    total_time += stage_four_time                           #and add to cumulative total time taken
    print("\nTime for stage 4:",                            #Print both calculated above values to user
          round(stage_four_time, 2),
          "seconds", round(total_time, 2), "total time")



if __name__ == "__main__":                                  #Only run script if called directly (i.e. not
    main()                                                  #as module in other program)