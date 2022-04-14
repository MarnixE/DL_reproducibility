from turtle import forward
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.optim as optim
import numpy as np
import pywt

# from haars_wavelet import haar_wavelet


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

# TCN block
class TCN_block(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        super(TCN_block, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)  # normal and std
        self.conv2.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


# TCN stacked blocks 
class TCN_net(nn.Module):
    def __init__(self, input_channels, n_channels, kernel_size, stride, dropout):
        super(TCN_net, self).__init__()
        layers = []
        num_levels = len(n_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_channels if i == 0 else n_channels[i - 1]
            out_channels = n_channels[i]
            layers += [TCN_block(in_channels,
                                 out_channels,
                                 kernel_size,
                                 stride,
                                 dilation=dilation_size,
                                 padding=(kernel_size - 1) * dilation_size,
                                 dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)






# Complete TCN Model
class TCN(nn.Module):
    def __init__(self, input_channels, n_channels, kernel_size, stride, dropout, n_outputs):
        super(TCN, self).__init__()
        
        # Create TCN blocks
        self.tcn = TCN_net(input_channels, n_channels, kernel_size, stride, dropout)

        # Init batch normalization
        self.input_bn = nn.BatchNorm1d(n_channels[-1])

        # Init final linear layer
        self.linear = nn.Linear(n_channels[-1], n_outputs)

        # Wavelet FC layer
        self.wv_linear = nn.Linear(n_channels[-1]*2, n_outputs)

    def haar_wavelet(data):
        col = data.shape[1]
        # print("col", col)
        row = data.shape[0]
        # print("col", row)
        # d3= data.shape[0]
        # print("col", d3)
        # finaldata= np.empty((d3,row*2,col))
        print(data.shape)


        for i in range(row):
            new_row = data[i,:].copy()
            # print(new_row.shape)
            (cA, cD) = pywt.dwt(new_row, 'haar')
            # print(new_col.shape)
            # print(cA.shape)
            # print(cD.shape)
            new_data1 = np.concatenate((cA,cD),0)
            # print(new_data1.shape)
            new_data2 = np.reshape(new_data1,(-1,col))
            print(new_data2.shape)
            new_data = np.vstack((data,new_data2))
            print(new_data.shape)

        return new_data


    def forward(self, x):
        y = self.tcn(x)
        y_norm = self.input_bn(y[:, :, -1])
        output = self.linear(y_norm)

        
        haar_input = x.cpu().detach().numpy()
        print(haar_input.shape)

        haar_features = self.haar_wavelet(haar_input)
        print(haar_features.shape)

        haar_features= torch.from_numpy(haar_features)
        haar_out = self.wv_linear(haar_features)

        return output, haar_out