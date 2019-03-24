import pandas as pd
import os

location = "C:\\Users\\Dan\\Dropbox\\Uni stuff\\EE3P\\"     #Location of source .wav and annotation file
lm_location = "C:\\Users\\Dan\\Documents\\LMs\\"            #Location of smaller .wavs from LM extractor
target_location = "C:\\Users\\Dan\\Documents\\LMlabels\\"   #Location to place new .lab files

for file in os.listdir(target_location):                    #Removes files already in LM directory
    os.remove(target_location + file)

annotation_file = pd.read_csv(                              #Annotation file imported as object, only the
    location + "siegfried_annotations.csv").iloc[195:666]   #rows pertaining to our source .wav file
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


for file_name in os.listdir(lm_location):                   #For each smaller .wav file
    f = open(target_location + file_name[:-4] +             #create a file with the same name but
             ".lab", "w+")                                  #as a .lab rather than a .wav
    start_time = int(file_name.split("_")[1].split("to")    #Get start time of small .wav in context
                     [0]) * 10000000                        #of overall source .wav time
    finish_time = int(file_name.split("_")[1].split("to")   #Get finish time of small .wav in context
                      [1]) * 10000000                       #of overall source .wav time
    true_start_time = 0                                     #Set start time for .lab in relation to
    num_allowed_lms = 0                                     #source wav and the number of allowed LMs
                                                            #in .lab file both to 0
    sfn = file_name.split(".wav")[0].split("_")[2:]         #Info on LMs (e.g. 'Nib19' or 'Hor_Wor2'
                                                            #is extracted from file name, stored as
                                                            #list of these (e.g. ['Hor', 'Wor2']
    for lm in sfn:                                          #For each different LM in the file name
        if any(i.isdigit() for i in lm):                    #if any contain a digit (e.g. 'Wor2')
            digit = int(lm[-1])                             #extract the digit as a integer,
            num_allowed_lms += digit                        #add this digit to number of allowed LMs
            if any(i.isdigit() for i in lm[:-1]):           #in file (accounting for number in '10s'
                digit_ten = int(lm[-2:-1]) * 10             #as well as single digits
                num_allowed_lms += digit_ten
        else:                                               #if it doesn't have a digit in name (e.g.
            num_allowed_lms += 1                            #'Hor'), add 1 to number of allowed LMs

    i = 0                                                   #Set iterator variable to '0'
    for st, ft, lbl in zip(start_times,                     #For each start time, finish time, LM label
                           finish_times, lm_labels):        #in the annotation file for each LM
        cts = time_converter([st, ft])                      #convert start/finish times to ms (as int)
        if int(cts[0]/10000000)*10000000 >= \
                start_time:                                 #If the start and finish times are
            if (int(cts[1]/10000000)*10000000) <= \
                    finish_time:                            #within the time the smaller .wav represents,
                if i == 0:                                  #if its the first of these allowed in the file,
                    if cts[0] > start_time:                 #and if its greater than the start time
                        f.write("0 \t" +                    #of the .lab file, write it to file as '0 \t
                                str(cts[0]-start_time) +    #<end of LM local to the file> \t <LM label>
                                "\t" + "Audio" + "\n")
                        true_start_time = start_time        #Update the start time relative to source file
                if num_allowed_lms > 0:                     #If there are still more LMs allowed to be
                    cts[0] = cts[0] - true_start_time       #written to file, calculate start/finish times
                    cts[1] = cts[1] - true_start_time       #relative to current .lab file and write
                    f.write(str(cts[0]) + "\t" +            #to file as <start LM time> \t <end LM time>
                            str(cts[1]) + "\t" +            #\t <LM label.
                            str(lbl)[2:-2] + "\n")
                    num_allowed_lms -= 1                    #Subtract from remaining allowed LMs into
                i += 1                                      #lab file, and increment iterator
    f.close()                                               #Close the file

    f = open(target_location +                              #Reopen the same .lab file for writing
             file_name[:-4] + ".lab", "r")
    lines = f.readlines()                                   #Read in the lines of the file as a list
    f.close()                                               #close the .lab file
    lines_to_insert = {}                                    #create empty dictionary of lines to insert
    for i in range(len(lines)):                             #For each line in the list of lines,
        if i != 0:                                          #if its not the first line,
            pos_1 = int(lines[i - 1].split("\t")[1])        #get the finish time of the previous line
            pos_2 = int(lines[i].split("\t")[0])            #and the start time of the current line
            if pos_2 > pos_1:                               #If there is a gap between the two times,
                line_to_insert = str(pos_1) + "\t" \
                                 + str(pos_2) + "\t" \
                                 + "Audio" + "\n"           #create an 'Audio' line to go between them
                lines_to_insert[line_to_insert] = i         #and add to dictionary with line to insert
                                                            #as the key and the position to insert as value
    offset = 0                                              #set list offset to 0
    for line in lines_to_insert:                            #For each line in the lines to insert dict
        lines.insert(lines_to_insert[line]+offset, line)    #insert line into the list of lines at correct
        offset += 1                                         #position, and increment the offset
    if lines:                                               #If the list of lines is not empty,
        if int(lines[-1].split("\t")[1]) < finish_time:     #if the last line's finish time is less than
                                                            #the file's finishing time,
            line_to_insert = str(lines[-1].split("\t")      #add an 'Audio' line that fills this gap
                                 [1]) + "\t" + \
                             str(finish_time-start_time) \
                             + "\t" + "Audio" + "\n"        #as <end time of last line> \t <end time of file>
                                                            #\t 'Audio'
            lines.insert(len(lines), line_to_insert)        #and insert this into lines at correct position
    f = open(target_location + file_name[:-4]               #Reopen .lab file for writing,
             + ".lab", "w")
    lines = "".join(lines)                                  #join these lines in the list as one string,
    f.write(lines)                                          #write these to the .lab file,
    f.close()                                               #and close the .lab file