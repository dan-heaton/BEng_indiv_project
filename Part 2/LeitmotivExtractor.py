from pydub import AudioSegment as AS
import pandas as pd
import os

location = "C:\\Users\\Dan\\Dropbox\\Uni stuff\\EE3P\\"     #Location of source .wav and annotation file
source_file = AS.from_wav(location +
                          "boulez_siegfried-act1.wav")      #Source file imported as object from directory
annotation_file = pd.read_csv(				 #Annotation file imported as object, only the
    location + "siegfried_annotations.csv").iloc[195:666]   #rows pertaining to our source .wav file
target_location = "C:\\Users\\Dan\\Documents\\LMs\\"        #Location to place newly created .wav files

for file in os.listdir(target_location):                    #Removes files already in LM directory
    os.remove(target_location + file)

start_times = annotation_file.iloc[:, 6:7].values           #Extract start times, finish times, LM labels
finish_times = annotation_file.iloc[:, 7:8].values          #from respective columns in annotation file object
lm_labels = annotation_file.iloc[:, 1:2].values


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


def reverse_time_converter(times):                          #Function takes in times as integers in a list,
    converted_times = []                                    #and converts each of them back to seconds,
    for time in times:                                      #rounding each down to closest second
        num_seconds = int(time / 1000)
        converted_times.append(num_seconds)
    return converted_times


def create_lm_file(window, lms):                            #Function takes in window pos' and LMs within
    label_strings = []
    prev_label = "EMP"
    for i in range(len(lms)):                               #for however many LMs to be in this file
        short_label = lms[i][2][:3]                         #shorten lbl for each LM
        if i != 0:                                          #if not the first LM in list of lms
            prev_label = label_strings[-1][1:]              #get previous label appended in list
        if short_label[:3] == prev_label[:3]:               #if prev lbl is same as current lbl
            if len(prev_label) == 3:                        #if len(prev label) is 3, only 1 of LM type,
                prev_label = "_" + prev_label + "2"         #so prev label changed to 'lbl2'
            else:                                           #else, last letter is incremented ('Jug4' to 'Jug5')
                prev_label = "_" + prev_label[:3] + \
                             str(int(prev_label[3:]) + 1)
            label_strings[-1] = prev_label                  #list appended with change
        else:
            label_strings.append("_" + short_label)         #if prev label not the same, append name on
                                                            #the end (e.g. 'Jug' becomes 'Jug_Nib')
    final_string = "".join(label_strings)                   #all strings in list are stuck together
    lm = source_file[window[0]:window[1]]                   #the smaller .wav is extracted from window pos'
    rcts = reverse_time_converter([window[0], window[1]])   #the window times converted to seconds
    lm.export(target_location + "boulSiegAct1mp3_" +        #smaller .wav written to target location with
              str(rcts[0]) + "to" + str(rcts[1]) +          #window times as seconds and string of LMs inside
              final_string + ".wav", format="wav")


window = [0, 40000]                                         #defines window as list with a start and end time
end_file_time = time_converter([finish_times[-1]])[0]       #gets end time from last time in annotation file
while(window[1] < end_file_time):                           #while the end of window not exceeded end of file
    lms = []                                                #empty LMs list defined with '0' current LM durat
    lm_duration = 0
    for st, ft, lbl in zip(start_times,                     #for each start time, finish time,
                           finish_times, lm_labels):        #label in annotation file
        cts = time_converter([st, ft])                      #convert start and finish times to integers
        lbl = str(lbl)[2:-2]                                #reduce lbl (e.g. "[[Mime]]" to "Mime")
        if (int(cts[0]) >= int(window[0])):                 #if LM start time greater than window start time
            if (int(cts[1]) <= int(window[1])):             #if LM end time less than window end time
                lms.append([cts[0], cts[1], lbl])           #add [st, ft, lbl] of LM to list of LMs
                lm_duration += (cts[1] - cts[0])            #add duration of LM to total durations
        elif (cts[0] >= window[0]) and \
                (cts[0] <= window[1]):                      #else, if only start time of LM within window
            window[1] = ft                                  #extend window to edge of LM
            lms.append([cts[0], cts[1], lbl])               #add [st, ft, lbl] of LM to list of LMs
            lm_duration += (cts[1] - cts[0])                #add duration of LM to total durations

    if len(lms) == 0:                                       #if no LMs extracted into list
        lm = source_file[window[0]:window[1]]               #create smaller .wav from window
        rcts = reverse_time_converter(                      #convert window times to milliseconds
            [window[0], window[1]])
        lm.export(target_location + "boulSiegAct1mp3_" +    #write .wav to file at target location
                  str(rcts[0]) + "to" + str(rcts[1]) +      #with times in seconds and '_aud' ending
                  "_aud.wav", format="wav")
    elif len(lms) == 1:                                     #else if there's 1 LM in the list
        create_lm_file(window, lms)                         #create smaller .wav from window and LMs list
    elif (lm_duration >= 3000) and (len(lms) >= 2):         #else if LM total duration exceeds 3 seconds and
                                                            #there are at least 2 LMs in the list
        create_lm_file(window, lms)                         #create smaller .wav from window and LMs list
    else:
        for st, ft, lbl in zip(start_times,                 #else, for each start time, finish time, lbl
                               finish_times, lm_labels):
            cts = time_converter([st, ft])                  #convert times to milliseconds
            lbl = str(lbl)[2:-2]                            #reduce lbl (e.g. "[[Mime]]" to "Mime")
            if cts[0] > lms[-1][0]:                         #if LM start time greater than finish time of
                                                            #last in list (i.e. occurs after last in list)
                lms.append([cts[0], cts[1], lbl])           #append this to list of LMs
                lm_duration += (cts[1] - cts[0])            #add duration of LM to total durations
                window[1] = cts[1]                          #extend window to end of LM
            if lm_duration >= 3000:                         #if LM list now contains more than 3 secs of LMs
                break                                       #then exit the loop
        create_lm_file(window, lms)                         #create smaller .wav from window and LMs list

    window[0] = window[1]                                   #Move along window so new start time is old end
    window[1] = window[0] + 40000                           #time and new end time is 4 seconds along