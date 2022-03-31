from data import DataExtraction
import pandas as pd
import torch
from IPython.display import display

class DataProcess(DataExtraction):
    def __init__(self):
        self.all_data = DataExtraction().extract()

    def train_data(self):
        x = self.all_data.iloc[:, 0:38].values
        y = self.all_data.iloc[:, 38].values

        self.x_train = torch.tensor(x)
        self.y_train = torch.tensor(y)

        return self.x_train, self.y_train


