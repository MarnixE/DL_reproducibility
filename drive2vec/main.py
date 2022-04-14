from model import tcn, triplet_loss, light_gb
from data import data_process
from utils import haars_wavelet
import pandas as pd
import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader , TensorDataset
import torch.optim as optim
import utils
import numpy as np
from IPython.display import display
from torchsummary import summary
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler 


torch.manual_seed(2020)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model parameters
n_inputs = 1000
input_size = 9
n_channels = [9, 9, 9, 9, 9]
n_outputs = 9
kernel_size = 16
stride = 1
dropout = 0.1






# summary(model, (38, 1000), device='cuda') # (in_channels, height, width)
# print(model)


# Split data
d_split = data_process.Data_preProcess()
drop = 5
anchor_train, pos_train, neg_train, anchor_test, pos_test, neg_test, input_size = d_split.get_split(drop)

n_channels = [input_size, input_size, input_size, input_size, input_size]
n_outputs = input_size
# Load model
model = tcn.TCN(input_size, n_channels, kernel_size, stride, dropout, n_outputs)
model.cuda()


d_train = data_process.DataProcess(anchor_train, pos_train, neg_train)
d_test = data_process.DataProcess(anchor_test ,pos_test ,neg_test)

train_loader = DataLoader(d_train, batch_size=65, shuffle=True)
test_loader = DataLoader(d_test, batch_size=20, shuffle=True)


train_features, train_pos, train_neg,  train_labels = next(iter(train_loader))

print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")


# def get_device():
#     if torch.cuda.is_available():
#         device = torch.device('cuda:0')
#     else:
#         device = torch.device('cpu') # don't have GPU 
#     return device

# device = get_device()

def evaluate_accuracy(data_loader, net, device=device):
    net.eval()  #make sure network is in evaluation mode

    #init
    acc_sum = torch.tensor([0], dtype=torch.float, device=device)
    n = 0

    for X, _, _, y in data_loader:
        # Copy the data to device.
        X, y = X.to(device, dtype=torch.float), y.to(device, dtype=torch.float)
        with torch.no_grad():
            y = y.long()
            acc_sum += torch.sum((torch.argmax(net(X), dim=1) == y))
            n += y.shape[0] #increases with the number of samples in the batch
    return acc_sum.item()/n


# Loss function and optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.jit.script(triplet_loss.TripletLoss())

# Classifier
clf = light_gb.lightGBClassifier(learning_rate=0.1)

# Training parameters
epochs = 100

train_accs = []
running_loss = []

def train_model():
    # Training loop
    for epoch in range(epochs):
        running_loss = []
        model.train()

        for step, (anchor_point, pos_point, neg_point, anchor_label) in enumerate(train_loader):
            
            anchor_point = anchor_point.to(device, dtype=torch.float)
            pos_point = pos_point.to(device, dtype=torch.float)
            neg_point = neg_point.to(device, dtype=torch.float)

            optimizer.zero_grad()

            anchor_out, haar_out = model(anchor_point)
            positive_out, _ = model(pos_point)
            negative_out, _ = model(neg_point)

            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.cpu().detach().numpy())

            clf_in = torch.cat((anchor_out, haar_out), 1)

            clf.classifier(clf_in, anchor_label[:, 0])
            

        print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, epochs, np.mean(running_loss)))
        print('')


train_model()

print("Training Done")
# # Save model
torch.save(model.state_dict(), "train_model.pth")


# model.load_state_dict(torch.load("train_model.pth"))

def test_model():
    for step, (anchor_point, pos_point, neg_point, anchor_label) in enumerate(test_loader):
            
            anchor_point = anchor_point.to(device, dtype=torch.float)
            pos_point = pos_point.to(device, dtype=torch.float)
            neg_point = neg_point.to(device, dtype=torch.float)


            anchor_out, haar_out = model(anchor_point)

            clf_in = torch.cat((anchor_out, haar_out), 1)
            clf_in = clf_in.cpu().detach().numpy()

            clf.predict(clf_in, anchor_label[:, 0])


test_model()