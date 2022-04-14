from data import DataExtraction
import pandas as pd
import torch
from IPython.display import display
import csv
import os
from torchvision import transforms
import numpy as np

class DataProcess(DataExtraction):
    def __init__(self):
        # self.all_data = DataExtraction().extract()
        current_path = os.getcwd()
        file_path_anchor = current_path + "/drive2vec/data/dataset/triplet_data/anchor0.csv"
        file_path_pos = current_path + "/drive2vec/data/dataset/triplet_data/positive0.csv"
        file_path_neg = current_path + "/drive2vec/data/dataset/triplet_data/negative0.csv"
        
        self.anchor_data = pd.read_csv(file_path_anchor)
        self.pos_data = pd.read_csv(file_path_pos)
        self.neg_data = pd.read_csv(file_path_neg)

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
            list_samples.pop(10)
            
            # Convert to 3D array
            split_array = np.dstack(list_samples)
            split_array = split_array.T

            return split_array

        array_anchor = self.anchor_data.to_numpy()
        array_pos = self.pos_data.to_numpy()
        array_neg = self.neg_data.to_numpy()

        # Split acnhor
        self.split_anchor = split_samples(array_anchor)


        # Split pos and neg
        array_pos = array_pos[0:18000, 1:10]
        split_pos = array_pos.reshape(18, -1, array_pos.shape[1])
        self.split_pos = np.transpose(split_pos, (0, 2, 1))

        array_neg = array_neg[0:18000, 1:10]
        split_neg = array_neg.reshape(18, -1, array_neg.shape[1])
        self.split_neg = np.transpose(split_neg, (0, 2, 1))


        

    def __len__(self):
        return len(self.split_anchor)

    def __getitem__(self, idx):
        x_anchor = self.split_anchor[idx, :9, :]
        x_pos = self.split_pos[idx, :9, :]
        x_neg = self.split_neg[idx, :9, :]
        y_anchor = self.split_anchor[idx, 9, :]

        # x_anchor = np.array(x_anchor, dtype=np.float64)
        
        # x_anchor = torch.Tensor(x_anchor)
        # x_pos = torch.Tensor(x_pos)
        # x_neg = torch.Tensor(x_neg)
        # y_anchor = torch.Tensor(y_anchor)

        # x_anchor = torch.from_numpy(x_anchor.to_numpy())
        # x_pos = torch.from_numpy(x_pos.to_numpy())
        # x_neg = torch.from_numpy(x_neg.to_numpy())
        # y_anchor = torch.from_numpy(y_anchor)

        return x_anchor, x_pos, x_neg, y_anchor





