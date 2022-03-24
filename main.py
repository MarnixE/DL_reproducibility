from data import data
from model import tcn
from utils import *

DE = data.DataExtraction()

X,Y,Z = DE.dataset_split()


model = tcn.TCN_net()

n_inputs = 32
n_channels = [32, 32, 32, 32, 32]
n_outputs = 32
kernel_size = 2
stride = 1
dropout = 0.1

net = tcn.TCN_net(n_inputs, n_channels, kernel_size, stride, dropout)



