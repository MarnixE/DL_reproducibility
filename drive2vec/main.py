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


torch.manual_seed(2020)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model parameters
n_inputs = 1000
input_size = 38
n_channels = [38, 38, 38, 38, 38]
n_outputs = 5
kernel_size = 3
stride = 1
dropout = 0.1

# Load model
model = tcn.TCN_net(input_size, n_channels, kernel_size, stride, dropout)
model.cuda()


# summary(model, (100, 39), device='cuda') # (in_channels, height, width)
# print(model)



# Load data
dp = data_process.DataProcess()
# train_data = dp.train_data()

# x_train_anchor, x_train_pos, x_train_neg, y_train_anchor = dp.train_data()
# train_data = [x_train_anchor, x_train_pos, x_train_neg, y_train_anchor]

# train_loader = DataLoader((x_train_anchor, x_train_pos, x_train_neg, y_train_anchor), batch_size=len(x_train_anchor), num_workers=5)

train_loader = DataLoader(dp, batch_size=3, num_workers=5)

# train_features, train_labels = next(iter(train_loader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")


# Loss function and optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.jit.script(triplet_loss.TripletLoss())

# Training parameters
learning_rate = 5e-1  # step size for gradient descent
epochs = 10  # how many times to iterate through the intire training set

model.train()

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu') # don't have GPU 
    return device

device = get_device()

for epoch in range(epochs):s
    model.train()
    running_loss = []
    for step, (anchor_point, pos_point, neg_point, anchor_label) in enumerate(train_loader):
        anchor_point = anchor_point.to(device, dtype=torch.float)
        pos_point = pos_point.to(device, dtype=torch.float)
        neg_point = neg_point.to(device, dtype=torch.float)
        
        optimizer.zero_grad()
        anchor_out = model(anchor_point)
        positive_out = model(pos_point)
        negative_out = model(neg_point)
        
        loss = criterion(anchor_out, positive_out, negative_out)
        # loss = criterion(anchor_out)
        loss.backward()
        optimizer.step()
        
        running_loss.append(loss.cpu().detach().numpy())
    print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, epochs, np.mean(running_loss)))








# # Define list to store loss of each iteration
# train_losses = []
# train_accs = []



# for i, (data, labels) in enumerate(trainloader):
#   print(data.shape, labels.shape)
#   print(data)
#   if i == 3:
#       break



# # Training loop
# model.train()
# for epoch in tqdm(range(epochs), desc="Epochs"):
#   running_loss = []
#   for step, (anchor_point, pos_point, neg_point, anchor_label) in enumerate(tqdm(train_loader, desc="Trainig", leave=False))

# # Training loop
# for epoch in range(epochs):
#     # Training loop
#     for i, (x_batch, y_batch) in enumerate(trainloader):
#         # Convert labels to one-hot encoding
#         y_batch = nn.functional.one_hot(y_batch, num_classes=10)

#         # Perform forward pass
#         y_pred = model.forward(x_batch)

#         loss, grad = tcn.CrossEntropyLoss(y_batch, y_pred)
#         train_losses.append(loss)
        
#         model.backward(grad)
#         model.optimizer_step(learning_rate)

#         # Calculate accuracy of prediction
#         correct = torch.argmax(y_pred, axis=1) == torch.argmax(y_batch, axis=1)
#         train_accs.append(torch.sum(correct)/len(y_pred))

#         # Print progress
#         if i % 200 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch + 1, i * len(x_batch), len(trainloader.dataset),
#                 100. * i / len(trainloader), loss))
