from data import DataExtraction
import pandas as pd
import torch
from IPython.display import display
import csv
import os
from torchvision import transforms
import numpy as np


class Data_preProcess(DataExtraction):
    def __init__(self):
        current_path = os.getcwd()
        file_path_anchor = current_path + "/driver2vec/data/dataset/triplet_data/anchor0.csv"
        file_path_pos = current_path + "/driver2vec/data/dataset/triplet_data/positive0.csv"
        file_path_neg = current_path + "/driver2vec/data/dataset/triplet_data/negative0.csv"
        
        self.anchor_data = pd.read_csv(file_path_anchor)
        self.pos_data = pd.read_csv(file_path_pos)
        self.neg_data = pd.read_csv(file_path_neg)

    def preprocess(self):
        def split_samples(in_array):
            list_samples = []

            split_idx = 0
            prev_user = 0
            prev_road = 0

            # Convert pandas to 3D numpy array (20*1000*39)
            for i in range (in_array.shape[0]):
                if (in_array[i, 39] != prev_user) or (in_array[i, 40] != prev_road):
                    sample_array = np.array(in_array[split_idx:i, 30:40])
                    list_samples.append(sample_array)
                    split_idx = i
                prev_user = in_array[i, 39]
                prev_road = in_array[i, 40]

            # Remove empty first entry and 10th because its size doesn't match (only 323 data points)
            list_samples.pop(0)
            # list_samples.pop(10)
            
            # Convert to 3D array
            split_array = np.dstack(list_samples)
            split_array = split_array.T

            return split_array

        array_anchor = self.anchor_data.to_numpy()
        array_pos = self.pos_data.to_numpy()
        array_neg = self.neg_data.to_numpy()

        # Split anchor
        self.split_anchor = split_samples(array_anchor)


        # Split pos and neg
        array_pos = array_pos[0:18000, 30:40]
        split_pos = array_pos.reshape(18, -1, array_pos.shape[1])
        self.split_pos = np.transpose(split_pos, (0, 2, 1))

        array_neg = array_neg[0:18000, 30:40]
        split_neg = array_neg.reshape(18, -1, array_neg.shape[1])
        self.split_neg = np.transpose(split_neg, (0, 2, 1))
    
    def get_split(self):
        self.preprocess()

        anchor = self.split_anchor
        positive = self.split_pos
        negative = self.split_neg

        # 18*10*1000
        # 90*10*200
        anchor_temp = np.zeros((90,10,200))
        p_temp = np.zeros((90,10,200))
        n_temp = np.zeros((90,10,200))
        for i in range(5):
            for j in range(18):
                anchor_temp[i*j,:,:] = anchor[j,:,i*200:(i+1)*200]
                p_temp[i*j,:,:] = positive[j,:,i*200:(i+1)*200]
                n_temp[i*j,:,:] = negative[j,:,i*200:(i+1)*200]


        anchor = anchor_temp
        positive = p_temp
        negative = n_temp
        # anchor = anchor.reshape(180, 1000)
        # positive = positive.reshape(180, 1000)
        # negative = negative.reshape(180, 1000)
        #
        # anchor = anchor.reshape(90, 10, 200)
        # positive = positive.reshape(90, 10, 200)
        # negative = negative.reshape(90, 10, 200)


        indices = np.random.permutation(anchor.shape[0])
        train_idx, test_idx = indices[:70], indices[70:]

        anchor_train = anchor[train_idx, :, :]
        pos_train = positive[train_idx, :, :]
        neg_train = negative[train_idx, :, :]

        anchor_test = anchor[test_idx, :, :]
        pos_test = positive[test_idx, :, :]
        neg_test = negative[test_idx, :, :]


        return(anchor_train, pos_train, neg_train, anchor_test, pos_test, neg_test)




class DataProcess(DataExtraction):
    def __init__(self, anchor, positive, negative):
        self.anchor = anchor
        self.pos = positive
        self.neg = negative
     

    def __len__(self):
        return len(self.anchor)

    def __getitem__(self, idx):
        x_anchor = self.anchor[idx, :9, :]
        x_pos = self.pos[idx, :9, :]
        x_neg = self.neg[idx, :9, :]
        y_anchor = self.anchor[idx, 9, :]

        print(y_anchor)

        return x_anchor, x_pos, x_neg, y_anchor





