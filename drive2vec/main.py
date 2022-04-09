from model import tcn, triplet_loss
from data import data_process
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader , TensorDataset
import torch.optim as optim
import utils
# from tqdm import tqdm
import numpy as np
from IPython.display import display
from torchsummary import summary
import matplotlib.pyplot as plt

torch.manual_seed(2020)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model parameters
n_inputs = 1000
input_size = 38
n_channels = [38, 38, 38, 38, 38]
n_outputs = 38
kernel_size = 16
stride = 1
dropout = 0.1

# Load model
model = tcn.TCN_net(input_size, n_channels, kernel_size, stride, dropout)
model.cuda()


# summary(model, (38, 1000), device='cuda') # (in_channels, height, width)
# print(model)


# Load data
dp = data_process.DataProcess()
train_loader = DataLoader(dp, batch_size=18, shuffle=True)

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
optimizer = optim.Adam(model.parameters(), lr=0.05)
criterion = torch.jit.script(triplet_loss.TripletLoss())

# Training parameters
epochs = 10

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

            anchor_out = model(anchor_point)
            positive_out = model(pos_point)
            negative_out = model(neg_point)
            
            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()
            
            running_loss.append(loss.cpu().detach().numpy())

        # train_acc = 100*evaluate_accuracy(train_loader, model.to(device))
        # train_accs.append(train_acc)
        print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, epochs, np.mean(running_loss)))
        # print('Accuracy of train set: {:.00f}%'.format(train_acc))
        print('')



train_model()

# Save model
torch.save({"model_state_dict": model.state_dict(),
            "optimzier_state_dict": optimizer.state_dict()
           }, "trained_model.pth")



# # Evaluate model
# train_results = []
# labels = []

# model.eval()
# with torch.no_grad():
#     for anchor, _, _, label in train_loader:
#         train_results.append(model(anchor.to(device, dtype=torch.float)).cpu().numpy())
#         labels.append(label)
        
# train_results = np.concatenate(train_results)
# labels = np.concatenate(labels)
# train_results.shape

# print(labels)

# plt.figure(figsize=(15, 10), facecolor="azure")
# for label in np.unique(labels):
#     tmp = train_results[labels==label]
#     plt.scatter(tmp[:, 0], tmp[:, 1], label=label)

# plt.legend()
# plt.show()