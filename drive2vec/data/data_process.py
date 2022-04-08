from data import DataExtraction
import pandas as pd
import torch
from IPython.display import display
import csv
import os
from torchvision import transforms

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

        self.x_anchor = self.anchor_data.iloc[:, 0:38].values
        self.y_anchor = self.anchor_data.iloc[:, 38].values

        self.x_pos = self.pos_data.iloc[:, 0:38].values

        self.x_neg = self.neg_data.iloc[:, 0:38].values

    def __len__(self):
        return len(self.x_anchor)

    # def __getitem__(self, item):
    #     self.x_anchor = torch.Tensor(self.x_anchor[item])
    #     self.x_pos = torch.Tensor(self.x_pos[item])
    #     self.x_neg = torch.Tensor(self.x_neg[item])
    #     self.y_anchor = torch.Tensor(self.y_anchor[item])

    #     return self.x_anchor, self.x_pos, self.x_neg, self.y_anchor

    def train_data(self):
        self.x_anchor = torch.Tensor(self.x_anchor)
        self.x_pos = torch.Tensor(self.x_pos)
        self.x_neg = torch.Tensor(self.x_neg)
        self.y_anchor = torch.Tensor(self.y_anchor)


    
        

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


