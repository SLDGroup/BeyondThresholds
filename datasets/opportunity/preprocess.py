import pandas as pd
import os
import numpy as np
import re
from utils.setup_funcs import PROJECT_ROOT

# ============================= constants =============================

body_parts = ["BACK",
              "RUA",
              "RLA",
              "LUA",
              "LLA",
              "L-SHOE",
              "R-SHOE"]

label_map = {0:"Null", 
             1:"Stand", 
             2:"Walk", 
             3:"Sit", 
             4:"Lie"
             }

label_col = "Locomotion"
sensor = 'InertialMeasurementUnit'

upper_channel_list = ["accX",
                      "accY",
                      "accZ",
                      "gyroX",
                      "gyroY",
                      "gyroZ"]

foot_channel_list = ["Body_Ax",
                     "Body_Ay",
                     "Body_Az",
                     "AngVelBodyFrameX",
                     "AngVelBodyFrameY",
                     "AngVelBodyFrameZ"]

sampling_rate = 30
window_len = 60 # 2 seconds
overlap_frac = 0.5

# there are 4 users
NUM_PARTICIPANTS = 4
# there are 5 normal runs and one drill run
NUM_RUNS = 6

# train-val-test split
train_runs = ["ADL1",
              "ADL2",
              "ADL3",
              "ADL4",
              "Drill"] # 0,1,2,3,5

val_runs = ["ADL5"] # 4

# load dataset structure
root_dir = os.path.join(PROJECT_ROOT,'raw_data/opportunity/OpportunityUCIDataset/dataset')

def get_column_mapping():
    '''
    This function returns a nested dictionary that returns all the columns
    and their corresponding sub items, e.g. InertialMeasurementUnit --> L-SHOE --> Body_Ax
    '''
    with open(os.path.join(root_dir,'column_names.txt'), 'r') as file:
        text = file.read()

    # Extract the strings between "Column:" and newline character or semicolon
    pattern = r'Column:\s+(\S.*?)(?=\n|;|$)'
    columns = re.findall(pattern, text)

    # Split the extracted strings into lists of individual words
    columns_list = [column.split() for column in columns]

    ms_idx = 0
    channel_idx_start = 1
    channel_idx_end = 243
    label_idx_start = 243

    # The "MILLISEC" column
    col_mapping_dict = {columns_list[ms_idx][1] : 0}

    # The sensor channel columns
    for col in columns_list[channel_idx_start:channel_idx_end]:
        col_idx = int(col[0]) - 1 # e.g. 0
        sensor = col[1] # e.g. "Accelerometer"
        position = col[2] # e.g. "RKN^"
        channel_type = col[3] # e.g. "accX" --> REED has an additional subchannel but we don't use REED so ignore

        # check if created sensor sub_dict
        if sensor not in col_mapping_dict.keys():
            col_mapping_dict[sensor] = {position : {channel_type : col_idx} }
        # check if created position sub_dict
        if position not in col_mapping_dict[sensor].keys():
            col_mapping_dict[sensor][position] = {channel_type : col_idx}
        else:
            col_mapping_dict[sensor][position][channel_type] = col_idx

    # The label columns
    for col in columns_list[label_idx_start:]:
        col_idx = int(col[0]) - 1 # e.g. 0
        label_level = col[1] # e.g. "Locomotion"
        col_mapping_dict[label_level] = col_idx

    return col_mapping_dict




# ============================= Preprocessing =============================

if __name__ == '__main__':

    # get the col idxs of desired channels
    col_map_dict = get_column_mapping()

    active_col_idxs = []
    for bp in body_parts:
        if bp == 'L-SHOE' or bp == 'R-SHOE':
            channel_list = foot_channel_list
        else:
            channel_list = upper_channel_list
        for ch in channel_list:
            active_col_idxs.append(col_map_dict[sensor][bp][ch])
    active_col_idxs.append(col_map_dict[label_col])
    active_col_idxs = np.array(active_col_idxs)

    # we separate the data by participant
    training_data = {user: [] for user in range(NUM_PARTICIPANTS)} # raw data
    training_labels = {user: [] for user in range(NUM_PARTICIPANTS)} # raw labels
    training_window_idxs = {user: [] for user in range(NUM_PARTICIPANTS)} # window start and stop
    training_window_labels = {user: [] for user in range(NUM_PARTICIPANTS)} # label for each window
    window_partitions = {user: [] for user in range(NUM_PARTICIPANTS)} # 0 for train, 1 for validation

    # keep track of list indices as accumulate data from each file
    curr_train_window_idxs = {user: 0 for user in range(NUM_PARTICIPANTS)}


    # for each participant, split their activity data into temporally disjoint
    # parts for training and testing and merge into body part specific arrays
    for user_i in range(NUM_PARTICIPANTS):
        # iterate through all runs
        for run_i in range(NUM_RUNS):

            # create the file name
            if run_i < 5:
                file_name = f"S{user_i+1}-ADL{run_i+1}.dat"
            else:
                file_name = f"S{user_i+1}-Drill.dat"
            data_file_path = os.path.join(root_dir,file_name)
            data_array = pd.read_csv(data_file_path, sep=' ', header=None).values

            print(f"file: {file_name}, curr: {curr_train_window_idxs}")

            # only keep the accelerometer/gyro columns for the desired body parts
            data_array = data_array[:,active_col_idxs]
            
            # remove rows with Nans
            non_nan_rows = (np.isnan(data_array).sum(axis=1) == 0).nonzero()[0]
            data_array = data_array[non_nan_rows]

            # split data and labels
            label_array = data_array[:,-1] # last columns
            data_array = data_array[:,:-1] # drop last column

            # map labels to be contiguous
            label_array[label_array == 4] = 3
            label_array[label_array == 5] = 4

            # form windows
            slide = int(window_len*(1-overlap_frac))
            start_idxs = np.concatenate([np.array([curr_train_window_idxs[user_i]]),
                                        np.arange(curr_train_window_idxs[user_i]+slide,
                                        curr_train_window_idxs[user_i]+data_array.shape[0]-window_len,
                                        slide)])
        
            end_idxs = start_idxs + window_len

            # need to set the window label to the most common one
            train_window_labels = []
            offset = start_idxs[0]
            for tsi,tei in zip(start_idxs-offset,end_idxs-offset):
                raw_labels = np.int64(label_array[tsi:tei])
                try:
                    train_window_labels.append(np.argmax(np.bincount(raw_labels)))
                except:
                    print(tsi,tei,len(start_idxs),label_array.shape)
                    exit()
            train_window_labels = np.array(train_window_labels)
            
            win_partitions = np.zeros(len(start_idxs))
            if run_i == 4:
                win_partitions[:] = 1 # validation windows for this activity, no overlap
            else:
                win_partitions[:] = 0
            curr_train_window_idxs[user_i] += data_array.shape[0]

            # put into list
            training_data[user_i].append(data_array)
            training_labels[user_i].append(label_array)

            training_window_idxs[user_i].append(np.stack([start_idxs,end_idxs]).T)
            training_window_labels[user_i].append(train_window_labels)
            window_partitions[user_i].append(win_partitions)

    os.mkdir(os.path.join(root_dir,"../../preprocessed_data"))

    # now concatenate, rescale, and save data
    for user_i in range(NUM_PARTICIPANTS):
        training_data[user_i] = np.concatenate(training_data[user_i])
    
        training_labels[user_i] = np.concatenate(training_labels[user_i])
        training_window_idxs[user_i] = np.concatenate(training_window_idxs[user_i])
        training_window_labels[user_i] = np.concatenate(training_window_labels[user_i])
        window_partitions[user_i] = np.concatenate(window_partitions[user_i])

        print(f"==== {user_i} ====")
        print(training_data[user_i].shape)
        print(training_labels[user_i].shape)
        print(training_window_idxs[user_i].shape)
        print(training_window_labels[user_i].shape)
        print(window_partitions[user_i].shape)

        folder = f"{root_dir}/../../preprocessed_data"
        np.save(f"{folder}/data_{user_i}",training_data[user_i])
        np.save(f"{folder}/labels_{user_i}",training_labels[user_i])
        np.save(f"{folder}/window_idxs_{user_i}",training_window_idxs[user_i])
        np.save(f"{folder}/window_labels_{user_i}",training_window_labels[user_i])
        np.save(f"{folder}/window_partitions_{user_i}",window_partitions[user_i])
