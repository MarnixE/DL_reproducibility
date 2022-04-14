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
        file_path_anchor = current_path + "/data/dataset/triplet_data/anchor0.csv"
        file_path_pos = current_path + "/data/dataset/triplet_data/positive0.csv"
        file_path_neg = current_path + "/data/dataset/triplet_data/negative0.csv"
        
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
                    sample_array = np.array(in_array[split_idx:i, 1:40])
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
        array_pos = array_pos[0:18000, 1:40]
        split_pos = array_pos.reshape(18, -1, array_pos.shape[1])
        self.split_pos = np.transpose(split_pos, (0, 2, 1))

        array_neg = array_neg[0:18000, 1:40]
        split_neg = array_neg.reshape(18, -1, array_neg.shape[1])
        self.split_neg = np.transpose(split_neg, (0, 2, 1))

    def drop_features(self):
        self.drop = {}
        self.drop[0] = [1, 2, 3, 4]
        self.drop[1] = [8, 9, 10, 11, 12, 13]
        self.drop[2] = [6, 19]
        self.drop[3] = [14, 24, 25, 26, 27, 28]
        self.drop[4] = [5, 6]
        self.drop[5] = [7, 38, ]
        self.drop[6] = [33, 34, 35, 36, 37]
        self.drop[7] = [22, 23]
        self.drop[8] = [21, 19]
    
    def get_split(self, d):
        self.preprocess()

        anchor = self.split_anchor
        positive = self.split_pos
        negative = self.split_neg

        # 18*10*1000
        # 90*10*200
        anchor_temp = np.zeros((90,39,200))
        p_temp = np.zeros((90,39,200))
        n_temp = np.zeros((90,39,200))
        for i in range(5):
            for j in range(18):
                anchor_temp[i*j,:,:] = anchor[j,:,i*200:(i+1)*200]
                p_temp[i*j,:,:] = positive[j,:,i*200:(i+1)*200]
                n_temp[i*j,:,:] = negative[j,:,i*200:(i+1)*200]

        anchor = anchor_temp
        positive = p_temp
        negative = n_temp


        indices = np.random.permutation(anchor.shape[0])
        train_idx, test_idx = indices[:70], indices[70:]

        anchor_train = anchor[train_idx, :, :]
        pos_train = positive[train_idx, :, :]
        neg_train = negative[train_idx, :, :]

        anchor_test = anchor[test_idx, :, :]
        pos_test = positive[test_idx, :, :]
        neg_test = negative[test_idx, :, :]

        self.drop_features()

        temp = np.arange(0,39)
        idx_ = []

        if d != -1:
            for i in temp:
                if i not in self.drop[d]:
                    idx_.append(i)

            anchor_train = anchor_train[:, idx_, :]
            pos_train = pos_train[:, idx_, :]
            neg_train = neg_train[:, idx_, :]

            anchor_test = anchor_test[:, idx_, :]
            pos_test = pos_test[:, idx_, :]
            neg_test = neg_test[:, idx_, :]

        size = anchor_train.shape[1] - 1
        return(anchor_train, pos_train, neg_train, anchor_test, pos_test, neg_test, size)




class DataProcess(DataExtraction):
    def __init__(self, anchor, positive, negative):
        self.anchor = anchor
        self.pos = positive
        self.neg = negative


     

    def __len__(self):
        return len(self.anchor)

    def __getitem__(self, idx):
        x_anchor = self.anchor[idx, :-1, :]
        x_pos = self.pos[idx, :-1, :]
        x_neg = self.neg[idx, :-1, :]
        y_anchor = self.anchor[idx, -1, :]

        return x_anchor, x_pos, x_neg, y_anchor





