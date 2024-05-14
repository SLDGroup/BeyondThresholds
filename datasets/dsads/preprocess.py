import numpy as np
import pandas as pd
import os
import re
from utils.setup_funcs import PROJECT_ROOT

# ============================= constants =============================
body_parts = ['torso',
              'right_arm',
              'left_arm',
              'right_leg',
              'left_leg']

label_map = {0:'sitting',
             1:'standing',
             2:'lying on back',
             3:'lying on right side',
             4:'ascending stairs',
             5:'descending stairs',
             6:'standing in elevator',
             7:'moving in elevator',
             8:'walking in parking lot',
             9:'walking on flat treadmill',
             10:'walking on inclined treadmill',
             11:'running on treadmill,',
             12:'exercising on stepper',
             13:'exercising on cross trainer',
             14:'cycling on exercise bike horizontal',
             15:'cycling on exercise bike vertical',
             16:'rowing',
             17:'jumping',
             18:'playing basketball'
            }

sensor_channels = ['accX',
                   'accY',
                   'accZ',
                   'gyroX',
                   'gyroY',
                   'gyroZ',
                   'magX',
                   'magY',
                   'magZ']

train_frac = 0.8
val_frac = 0.2 # last 20% of data from each training user
# test will be leave one person group out cross validation

sampling_rate = 25
window_len = 50 # (2 seconds)
overlap_frac = 0.5

NUM_PARTICIPANTS = 8
NUM_ACTIVITIES = 19
SEGMENT_LEN = 125 # samples per segment file, 5 seconds * 25 Hz 
NUM_SEGMENTS = 60 # 5 minutes (300 seconds) / 5 second segments
NUM_SENSOR_CHANNELS = 45 # 5 body parts * 9 channels
ACC_GYRO_CHANNELS = np.array([0,1,2,3,4,5,
                              9,10,11,12,13,14, 
                              18,19,20,21,22,23, 
                              27,28,29,30,31,32, 
                              36,37,38,39,40,41
                            ])

num_samples_per_activity = SEGMENT_LEN*NUM_SEGMENTS


# ============================= Preprocessing =============================

if __name__ == '__main__':

    # load dataset structure
    root_dir = os.path.join(PROJECT_ROOT,'raw_data/dsads/data')
    activity_folders = os.listdir(root_dir)
    activity_folders.sort(key=lambda f: int(re.sub('\D', '', f))) # dont include other preprocessed data
    participant_folders = os.listdir(os.path.join(root_dir,activity_folders[0]))
    participant_folders.sort(key=lambda f: int(re.sub('\D', '', f)))
    segment_files = os.listdir(os.path.join(root_dir,activity_folders[0],participant_folders[0]))
    segment_files.sort(key=lambda f: int(re.sub('\D', '', f)))

    # we separate the data by participant
    training_data = {user: [] for user in range(NUM_PARTICIPANTS)} # raw data
    training_labels = {user: [] for user in range(NUM_PARTICIPANTS)} # raw labels
    training_window_idxs = {user: [] for user in range(NUM_PARTICIPANTS)} # window start and stop
    training_window_labels = {user: [] for user in range(NUM_PARTICIPANTS)} # label for each window
    window_partitions = {user: [] for user in range(NUM_PARTICIPANTS)} # 0 for train, 1 for validation

    # keep track of list indices as accumulate data from each file
    curr_train_window_idxs = {user: 0 for user in range(NUM_PARTICIPANTS)}
    file_count = {pf:0 for pf in participant_folders}

    min_val_acc = 0
    max_val_acc = 0
    min_val_gyro = 0
    max_val_gyro = 0

    # merge data for each participant
    for user_i, user_folder in enumerate(participant_folders):
        for activity_i, activity_folder in enumerate(activity_folders):
            print(f"user: {user_i}, activity: {label_map[activity_i]}, curr: {curr_train_window_idxs}")
            # create the data array which contains samples across all segment files
            data_array = np.zeros((num_samples_per_activity,len(ACC_GYRO_CHANNELS)))
            label_array = np.zeros(num_samples_per_activity)
            for segment_i, segment_file in enumerate(segment_files):
                data_file_path = os.path.join(root_dir,activity_folder,user_folder,segment_file)
                data_segment = pd.read_csv(data_file_path,header=None).values
                start = segment_i*SEGMENT_LEN
                end = start + SEGMENT_LEN
                data_array[start:end,:] = data_segment[:,ACC_GYRO_CHANNELS]
                label_array[start:end] = activity_i
            

            # form windows
            slide = int(window_len*(1-overlap_frac))
            start_idxs = np.concatenate([np.array([curr_train_window_idxs[user_i]]),
                                        np.arange(curr_train_window_idxs[user_i]+slide,
                                        curr_train_window_idxs[user_i]+data_array.shape[0]-window_len,
                                        slide)]) # [0+offset,25+offset,50+offset,...,7450+offset]
            
            # split into training, validation
            num_train_samples = int(len(start_idxs)*train_frac)
            num_val_samples = int(len(start_idxs)*val_frac)
        
            end_idxs = start_idxs + window_len
            train_window_labels = np.array([activity_i]*len(start_idxs))
            win_partitions = np.zeros(len(start_idxs))
            win_partitions[num_train_samples+1:] = 1 # validation windows for this activity, no overlap
            curr_train_window_idxs[user_i] += data_array.shape[0]

            # put into list
            training_data[user_i].append(data_array)
            training_labels[user_i].append(label_array)

            training_window_idxs[user_i].append(np.stack([start_idxs,end_idxs]).T)
            training_window_labels[user_i].append(train_window_labels)
            window_partitions[user_i].append(win_partitions)


    os.mkdir(os.path.join(root_dir,"../preprocessed_data"))

    # now concatenate and save data
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

        folder = f"{root_dir}/../preprocessed_data"
        np.save(f"{folder}/data_{user_i}",training_data[user_i])
        np.save(f"{folder}/labels_{user_i}",training_labels[user_i])
        np.save(f"{folder}/window_idxs_{user_i}",training_window_idxs[user_i])
        np.save(f"{folder}/window_labels_{user_i}",training_window_labels[user_i])
        np.save(f"{folder}/window_partitions_{user_i}",window_partitions[user_i])