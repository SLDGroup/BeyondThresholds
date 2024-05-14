import pandas as pd
import os
import numpy as np
import re
from utils.setup_funcs import PROJECT_ROOT

'''
A few notes
-for each segment we truncate the start and end samples which are noisy
-we concatenate the activity data files which are split (e.g. walk1, walk2)
'''

# ============================= constants =============================

body_parts = ['chest',
              'forearm',
              'head',
              'shin',
              'thigh',
              'upperarm',
              'waist']

activities = ['climbingdown',
              'climbingup',
              'jumping',
              'lying',
              'running',
              'sitting',
              'standing',
              'walking']

train_frac = 0.8
val_frac = 0.2

sampling_rate = 50
window_len = 100 # 2 seconds
overlap_frac = 0.5

NUM_PARTICIPANTS = 15


# ============================= Preprocessing =============================

if __name__ == '__main__':
    # load dataset structure
    root_dir = os.path.join(PROJECT_ROOT,'raw_data/rwhar/data')
    participant_folders = os.listdir(root_dir)
    participant_folders.sort(key=lambda f: int(re.sub('\D', '', f)))

    # we separate the data by participant
    training_data = {user: [] for user in range(NUM_PARTICIPANTS)} # raw data
    training_labels = {user: [] for user in range(NUM_PARTICIPANTS)} # raw labels
    training_window_idxs = {user: [] for user in range(NUM_PARTICIPANTS)} # window start and stop
    training_window_labels = {user: [] for user in range(NUM_PARTICIPANTS)} # label for each window
    window_partitions = {user: [] for user in range(NUM_PARTICIPANTS)} # 0 for train, 1 for validation

    # keep track of list indices as accumulate data from each file
    curr_train_window_idxs = {user: 0 for user in range(NUM_PARTICIPANTS)}
    file_count = {pf:0 for pf in participant_folders}

    active_users = []

    # first filter out users which have missing data
    for user_i,participant_folder in enumerate(participant_folders):
        # for a given user, get all activity csvs (8 activities, 7 body parts)
        activity_csvs = os.listdir(os.path.join(root_dir,participant_folder,'data'))
        activity_csvs.sort()
        print(participant_folder)

        # keep a dict of activity files present
        act_bp_dict = {act:{} for act in activities}
        for k in act_bp_dict.keys():
            for bp in body_parts:
                act_bp_dict[k][bp] = False

        # iterate over activity files
        for activity_csv in activity_csvs:
            if activity_csv.endswith(".csv"):
                # determine the label and body part
                activity_str = activity_csv.split("_")[1]
                body_part = activity_csv.split("_")[2]
                if body_part.isdigit(): # some files are split into parts
                    body_part = activity_csv.split("_")[3]
                file_count[participant_folder] += 1
                body_part = body_part[:-4]
                label = activities.index(activity_str)
                act_bp_dict[activity_str][body_part] = True

        # if have less than 8*7 True values, then data is missing
        count = 0
        for act in act_bp_dict.keys():
            for bp in act_bp_dict[act]:
                if act_bp_dict[act][bp] == True:
                    count += 1
                else:
                    print(f"User {participant_folder} is missing {act}-{bp}")
        if count == len(body_parts)*len(activities):
            active_users.append(user_i)
    print(f"active_users (idxs): {active_users}")

    # then load all the data
    for user_i,participant_folder in enumerate(participant_folders):
        if user_i not in active_users:
            print(f"Skipping {participant_folder}")
            continue

        # keep a dict of activity data present
        act_bp_dict = {act:{} for act in activities}
        for k in act_bp_dict.keys():
            for bp in body_parts:
                act_bp_dict[k][bp] = []

        # for a given user, get all activity csvs (8 activities, 7 body parts)
        activity_csvs = os.listdir(os.path.join(root_dir,participant_folder,'data'))
        activity_csvs.sort()
        print(participant_folder)

        # for each activity, get all body part csvs
        for activity_i,activity in enumerate(activities):
            prefix = f"acc_{activity}_"
            for activity_csv in activity_csvs:
                if activity_csv.startswith(prefix) and activity_csv.endswith(".csv"):
                    activity_str = activity_csv.split("_")[1]
                    body_part = activity_csv.split("_")[2]
                    if body_part.isdigit(): # some files are split into parts
                        body_part = activity_csv.split("_")[3]
                    # load the data for every body part
                    file_path = os.path.join(root_dir,participant_folder,'data',activity_csv)
                    print(f"{activity_str}-{body_part[:-4]}")
                    # filter start and end segments with no activity, don't need id from csv
                    if activity_str == 'jumping':
                        act_bp_dict[activity_str][body_part[:-4]].append(pd.read_csv(file_path).values[window_len:,1:])
                    else:
                        act_bp_dict[activity_str][body_part[:-4]].append(pd.read_csv(file_path).values[3*window_len:-3*window_len,1:])

        # first merge csvs for activities that got split into segments
        for act in act_bp_dict.keys():
            for bp in act_bp_dict[act]:
                if len(act_bp_dict[act][bp]) > 1:
                    data_arrays = act_bp_dict[act][bp]
                    act_bp_dict[act][bp] = np.concatenate(data_arrays,axis=0)
                else:
                    # no more list
                    act_bp_dict[act][bp] = act_bp_dict[act][bp][0]

        # now try to temporally align data across body parts as best as possible
        for act_i,act in enumerate(act_bp_dict.keys()):
            start_times = []
            for bp in act_bp_dict[act]:
                start_times.append(act_bp_dict[act][bp][0,0])
            print(f"{participant_folder}-{act}: {start_times}")
            latest_start = max(start_times)
            # remove initial rows if can get closer to the latest start time
            for bp in act_bp_dict[act]:
                times = act_bp_dict[act][bp][:,0]
                closest_idx = np.argmin(abs(times - latest_start))
                act_bp_dict[act][bp] = act_bp_dict[act][bp][closest_idx:,1:]
            # get min length so duration is the same
            lengths = []
            for bp in act_bp_dict[act]:
                lengths.append(act_bp_dict[act][bp].shape[0])
            print(f"{participant_folder}-{act}: {lengths}")
            shortest = min(lengths)
            for bp in act_bp_dict[act]:
                act_bp_dict[act][bp] = act_bp_dict[act][bp][:shortest,:]

            # now merge body parts into one array
            trunc_len = shortest-shortest%window_len
            data_array = np.zeros((trunc_len,3*len(body_parts)))
            label_array = np.zeros(trunc_len)

            for bp_i,bp in enumerate(act_bp_dict[act]):
                data_array[:,bp_i*3:(bp_i+1)*3] = act_bp_dict[act][bp][:trunc_len,:]
            label_array[:] = act_i
            print(data_array.shape)
            

            # now we can collect the windows
            # form windows
            slide = int(window_len*(1-overlap_frac))
            start_idxs = np.concatenate([np.array([curr_train_window_idxs[user_i]]),
                                        np.arange(curr_train_window_idxs[user_i]+slide,
                                        curr_train_window_idxs[user_i]+data_array.shape[0]-window_len,
                                        slide)]) # [0+offset,50+offset,100+offset,...]
            
            # split into training, validation
            num_train_samples = int(len(start_idxs)*train_frac)
            num_val_samples = int(len(start_idxs)*val_frac)

            end_idxs = start_idxs + window_len
            train_window_labels = np.array([act_i]*len(start_idxs))
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

    # now concatenate, rescale, and save data
    for user_i in range(NUM_PARTICIPANTS):
        if user_i not in active_users:
            print(f"Skipping {participant_folder}")
            continue
        print(len(training_data[user_i]))
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