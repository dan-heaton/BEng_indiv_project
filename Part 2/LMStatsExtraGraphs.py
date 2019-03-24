import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

location = "C:\\Users\\Dan\\Dropbox\\Uni stuff\\EE3P\\"     #Location of annotation file
annotation_file = pd.read_csv(                              #Annotation file imported as object, only the
    location + "siegfried_annotations.csv").iloc[195:666]   #rows pertaining to our source .wav file

start_times = annotation_file.iloc[:, 6:7].values           #Extract start times, finish times, LM labels,
finish_times = annotation_file.iloc[:, 7:8].values          #and salience ratings from respective
lm_labels = annotation_file.iloc[:, 1:2].values             #columns in annotation file object
ratings = annotation_file.iloc[:, 5:6].values

lm_type_count = {}                                          #setup empty dictionary for LM names and counts
lm_names = []                                               #names of LMs in dictionary
counts = []                                                 #counts of LMs in dictionary

for lm in lm_labels:                                        #For each LM label in annotation file
    lm = str(lm)[2:-2]                                      #reduce lbl (e.g. "[[Mime]]" to "Mime")
    if lm in lm_type_count:                                 #If an LM is already in dictionary,
        lm_type_count[lm] += 1                              #increment its count
    else:                                                   #else, create new LM in dict and set
        lm_type_count[lm] = 1                               #count to 1

od = OrderedDict(sorted(lm_type_count.items(),              #Sort dictionary by ascending counts
                        key=lambda t: t[1]))
for lm_name in od:                                          #For each LM label in sorted dictionary
    lm_names.append(lm_name)                                #append it to a list (full name)
    counts.append(od[lm_name])                              #and append its count to a list
lm_names = lm_names[::-1][:5]                               #Reverse both lists so counts are descending
counts = counts[::-1][:5]                                   #with LM label list matches positions
                                                            #and reducing lists to only first 5 (most common)


def time_converter(times):                                  #Function takes in times as list of strings,
    converted_times = []                                    #e.g. "[[01:13:45:384]]"
    for time in times:                                      #for each time in the list, convert to string,
        time = str(time)[2:-2]                              #remove ends of string to leave "01:13:45:384",
        split = str(time).split(':')                        #split each part on colons into list of parts,
        hrs_to_ms = int(split[0]) * 60 * 60 * 1000          #convert first part (hours) to milliseconds,
        mins_to_ms = int(split[1]) * 60 * 1000              #second part (minutes) to milliseconds,
        secs_to_ms = int(split[2]) * 1000                   #third part (seconds) to milliseconds,
        total_time = hrs_to_ms + mins_to_ms + \
                     secs_to_ms + (int(split[3]) * 10)      #and add all of them together to get time
        converted_times.append(total_time)                  #in milliseconds, and add to list
    return converted_times                                  #return converted times in milliseconds


for lm_name, count in zip(lm_names, counts):                #For each LM label and corresponding count
    durations = []                                          #create a new empty durations list
    for st, ft, lbl in zip(start_times, finish_times,       #For each start time, finish time, LM label
                           lm_labels):                      #in the annotation file,
        if lbl == lm_name:                                  #if the LM in the annot file matches the one
            cts = time_converter([st, ft])                  #we're working with currently, convert its
            duration = abs(cts[1] - cts[0]) / 1000          #start and end times to milliseconds as integer,
            durations.append(duration)                      #calculate duration from these in seconds and
                                                            #append to list of durations
    plt.hist(durations, 100)                                #Plot this histogram of durations
    plt.xlabel("'" + str(lm_name) +                         #Set x-axis based on name of LM we're looking at
               "' duration (seconds)")
    plt.ylabel("Frequency")                                 #Set y-axis to 'frequency'
    plt.title("Frequency of " + str(lm_name) +              #Set title based on name of LM we're looking at
              " durations (" + str(count) + " total)")      #and also its number of occurences in annot file
    plt.show()                                              #Display the graph