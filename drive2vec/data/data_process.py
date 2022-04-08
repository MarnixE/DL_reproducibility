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

        # self.x_anchor = self.anchor_data.iloc[:, 0:38].values
        # self.y_anchor = self.anchor_data.iloc[:, 38].values
        # self.x_pos = self.pos_data.iloc[:, 0:38].values
        # self.x_neg = self.neg_data.iloc[:, 0:38].values

    def __len__(self):
        return len(self.anchor_data)

    def __getitem__(self, idx):
        x_anchor = self.anchor_data.iloc[idx, 0:39]
        # x_pos = self.pos_data.iloc[idx, 0:39]
        # x_neg = self.neg_data.iloc[idx, 0:39]
        y_anchor = self.anchor_data.iloc[idx, 39]
        
        x_anchor = torch.Tensor(x_anchor)
        # x_pos = torch.Tensor(x_pos)
        # x_neg = torch.Tensor(x_neg)
        # y_anchor = torch.Tensor(y_anchor)

        # x_anchor = torch.from_numpy(x_anchor.to_numpy())
        # x_pos = torch.from_numpy(x_pos.to_numpy())
        # x_neg = torch.from_numpy(x_neg.to_numpy())
        # y_anchor = torch.from_numpy(y_anchor)

        return x_anchor, y_anchor

    # def train_data(self):
    #     self.x_anchor = torch.Tensor(self.x_anchor)
    #     # self.x_pos = torch.Tensor(self.x_pos)
    #     # self.x_neg = torch.Tensor(self.x_neg)
    #     self.y_anchor = torch.Tensor(self.y_anchor)


    
        

    # def train_data(self):

    #     self.x_anchor = []
    #     self.x_pos = []
    #     self.x_neg = []
    #     # self.y_anchor = []


    #     for i in range(self.x_anchor_in.shape[0]):
    #         self.x_anchor.append(torch.Tensor(self.x_anchor_in[i]))
    #         self.x_pos.append(torch.Tensor(self.x_pos_in[i]))
    #         self.x_neg.append(torch.Tensor(self.x_neg_in[i]))
       


    #     # self.x_train = torch.tensor(x)
    #     # self.y_train = torch.tensor(y)

    #     return self.x_anchor, self.x_pos, self.x_neg, self.y_anchor


