from model import tcn, triplet_loss
from data import data_process
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader , TensorDataset
import utils


# Model parameters
n_inputs = 32
n_channels = [32, 32, 32, 32, 32]
n_outputs = 32
kernel_size = 2
stride = 1
dropout = 0.1

# Load model
model = tcn.TCN_net(n_inputs, n_channels, kernel_size, stride, dropout)

# Load data
dp = data_process.DataProcess()
x_train, y_train = dp.train_data()

dataset = TensorDataset(x_train, y_train)
trainloader = DataLoader(dataset , batch_size = len(x_train), shuffle=False)

print(trainloader)

# Loss function
criterion = torch.jit.script(triplet_loss.TripletLoss())

# Training parameters
learning_rate = 5e-1  # step size for gradient descent
epochs = 10  # how many times to iterate through the intire training set

# Define list to store loss of each iteration
train_losses = []
train_accs = []

for i, (data, labels) in enumerate(trainloader):
  print(data.shape, labels.shape)
  print(data)
  if i == 3:
      break



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
