import torch
from torch.utils.data import Dataset
import os
import numpy as np
from utils.setup_funcs import PROJECT_ROOT


class RWHAR(Dataset):
    """ PyTorch dataset class for the preprocessed rwhar dataset

    Parameters
    ----------

    root_dir: str
        global path to the dsads preprocessed data

    users: list
        list of users to load data for, subset of [1,3,4,5,7,8,9,10,11,12,13,14,15]

    body_parts: list
        list of body parts to get sensor channels from, subset of ['chest','forearm','head','shin','thigh','upperarm','waist']

    train: bool
        whether to get the training data

    val: bool
        whether to get the validation data

    """
    def __init__(self, root_dir,users,body_parts,train=False,val=False):
        # activity labels (order matters)
        self.label_map = {0:'climbingdown',1:'climbingup',2:'jumping',3:'lying',4:'running',5:'sitting',6:'standing',7:'walking'}
        
        self.body_part_map = {'chest':[0,1,2],'forearm':[3,4,5],'head':[6,7,8],'shin':[9,10,11],
                              'thigh':[12,13,14],'upperarm':[15,16,17],'waist':[18,19,20]}
        
        self.train = train
        self.val = val
        prefix = f"{root_dir}/"

        self.users = users
        self.body_parts = []
        for bp in body_parts:
            self.body_parts.extend(self.body_part_map[bp])
        self.body_parts = np.array(self.body_parts)

        self.raw_data = [[] for user in users]
        self.raw_labels = [[] for user in users]
        self.window_idxs = [[] for user in users]
        self.window_labels = [[] for user in users]
        self.window_partitions = self.raw_data = [[] for user in users]

        # load the data
        for user_i, user in enumerate(self.users):
            self.window_partitions[user_i] = np.load(f"{prefix}window_partitions_{user-1}.npy") # (n_w)
            if train == True:
                idxs = (self.window_partitions[user_i] == 0).nonzero()[0]
            elif val == True:
                idxs = (self.window_partitions[user_i] == 1).nonzero()[0]
            else:
                idxs = np.arange(0,self.window_partitions[user_i].shape[0]) # all

            self.raw_data[user_i] = np.load(f"{prefix}data_{user-1}.npy")[:,self.body_parts] # (n,ch)
            self.raw_labels[user_i] = np.load(f"{prefix}labels_{user-1}.npy") # (n)
            self.window_idxs[user_i] = np.load(f"{prefix}window_idxs_{user-1}.npy")[idxs,:] # (n_w,2)
            self.window_labels[user_i] = np.load(f"{prefix}window_labels_{user-1}.npy")[idxs] # (n_w)

    def preprocess(self,rescale=None,normalize=False):
        """ rescaling and normalization using training statistics

        Parameters
        ----------

        rescale: list 
            [min,max] range to rescale

        normalize: bool
            whether to subtract mean and divide by standard deviation

        """

        # use training min,max on test data
        if rescale is not None:
            all_data = np.concatenate(self.raw_data)
            if self.train == True:
                self.min_val = np.min(all_data)
                self.max_val = np.max(all_data)
            for user_i, user in enumerate(self.users): 
                self.raw_data[user_i] = ((self.raw_data[user_i]-self.min_val)/(self.max_val-self.min_val))*(rescale[1]-rescale[0]) + rescale[0]
        
        # use train mean, std on test data
        if normalize == True:
            all_data = np.concatenate(self.raw_data)
            if self.train == True:
                self.mean = np.mean(all_data)
                self.std = np.std(all_data)
            for user_i, user in enumerate(self.users):
                self.raw_data[user_i] = (self.raw_data[user_i]-self.mean)/(self.std + 1e-5)

    def __getitem__(self, idx):
        # index into user then window
        # e.g. [(0,1000), (1000,2000)], if idx = 1200 then count is 1000 so idx becomes 200
        count = 0
        for user_i,user_windows in enumerate(self.window_idxs):
            count += user_windows.shape[0]
            if count > idx:
                count -= user_windows.shape[0]
                break
        
        # print(f"index: {idx}")
        idx = idx - count

        # get the window idxs
        start,end = self.window_idxs[user_i][idx]
        # print(f"idx: {idx}, count: {count}, user_i: {user_i}, start,end: {start},{end}")
        
        # get the data window
        X = self.raw_data[user_i][start:end,:]

        # get the label
        Y = self.window_labels[user_i][idx]

        # return the sample and the class
        # transpose because we want (C x L), i.e. each row is a new channel and columns are time
        return torch.tensor(X.T).float(), torch.tensor(Y).long()

    def __len__(self):
        return sum([len(wl) for wl in self.window_labels])
    

def load_rwhar_person_dataset(batch_size,train_users,test_users,body_parts,shuffle=True):
    root_dir = os.path.join(PROJECT_ROOT,'raw_data/rwhar/preprocessed_data')

    train_ds = RWHAR(root_dir,train_users,body_parts,train=True)
    train_ds.preprocess([-1,1],True)

    val_ds = RWHAR(root_dir,train_users,body_parts,val=True)
    val_ds.min_val = train_ds.min_val
    val_ds.max_val = train_ds.max_val
    val_ds.mean = train_ds.mean
    val_ds.std = train_ds.std
    val_ds.preprocess([-1,1],True)

    test_ds = RWHAR(root_dir,test_users,body_parts)
    test_ds.min_val = train_ds.min_val
    test_ds.max_val = train_ds.max_val
    test_ds.mean = train_ds.mean
    test_ds.std = train_ds.std
    test_ds.preprocess([-1,1],True)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, pin_memory=False,drop_last=True,num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=False,drop_last=True,num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=False,drop_last=True,num_workers=4)

    return train_loader, val_loader, test_loader