import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np

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
    lm_names.append(lm_name[:5])                            #append it to a list (first 5 letters)
    counts.append(od[lm_name])                              #and append its count to a list
lm_names = lm_names[::-1]                                   #Reverse both lists so counts are descending
counts = counts[::-1]                                       #with LM label list matches positions

plt.bar(range(len(lm_names)), counts, align='center')       #Plot bar graph as with names in LM label list
plt.xticks(range(len(lm_names)), lm_names, rotation=45)     #as x-axis, the bar's values as corresponding
plt.xlabel("Leitmotiv names")                               #positions in count list
plt.ylabel("Leitmotiv frequency")                           #Set the x labels as LM names at 45 degrees
plt.title("Frequency of specific leitmotivs")               #Set the x-axis, y-axis, and title labels
plt.show()                                                  #and display the graph

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

durations = []                                              #Create empty durations list
for st, ft in zip(start_times, finish_times):               #For each start/finish time in annot file
    cts = time_converter([st, ft])                          #Convert the strings to milliseconds in integer
    duration = abs(cts[1] - cts[0]) / 1000                  #Find difference in start/finish times in seconds
    durations.append(duration)                              #and add to durations list

plt.hist(durations, 100)                                    #Plot all durations of LMs in durations list
plt.xlabel("Leitmotiv duration (seconds)")                  #as histogram
plt.ylabel("Frequency")                                     #Set the x-axis, y-axis, and title labels
plt.title("Frequency of leitmotiv durations")
plt.axis([0, 25, 0, 120])                                   #Set the scale of the axes
plt.show()                                                  #and display the graph

final_ratings = []                                          #Create empty ratings list
for rating in ratings:                                      #Extract ratings from each list
    for r in rating:                                        #from the ratings column in annotation file
        final_ratings.append(int(r))                        #and append to ratings list

plt.scatter(durations, final_ratings)                       #Plot scatter plot with durations on x-axis
plt.plot(np.unique(durations),                              #and extracted salience ratings on y-axis
         np.poly1d(np.polyfit(durations, final_ratings,     #with a line of best fit running through them
                              1))(np.unique(durations)))
plt.xlabel("Leitmotiv duration (seconds)")                  #Set the x-axis, y-axis, and title labels
plt.ylabel("Salience rating (1-5)")
plt.title("Ratings of leitmotivs against duration")
plt.show()                                                  #Display the graph