import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.optim as optim


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


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


class TCN_net(nn.Module):
    def __init__(self, n_inputs, n_channels, kernel_size, stride, dropout):
        super(TCN_net, self).__init__()
        layers = []
        num_levels = len(n_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = n_inputs if i == 0 else n_channels[i - 1]
            out_channels = n_channels[i]
            layers += [TCN_block(in_channels,
                                 out_channels,
                                 kernel_size,
                                 stride,
                                 dilation=dilation_size,
                                 padding=(kernel_size - 1) * dilation_size,
                                 dropout=dropout)]

        self.network = nn.Sequential(*layers)


n_inputs = 32
n_channels = [32, 32, 32, 32, 32]
n_outputs = 32
kernel_size = 2
stride = 1
dropout = 0.1

net = TCN_net(n_inputs, n_channels, kernel_size, stride, dropout)

def CrossEntropyLoss(y_true, y_pred):
    
    # Calculate softmax using previously defined function
    softmax = Softmax(y_pred)

    # Convert one-hot vector to class id
    y_true = torch.argmax(y_true, axis=1)
    # Get number of samples in batch
    n = y_true.shape[0]
    # Calculate cross entropy loss between y_true and y_pred
    log_likelihood = -torch.log(y_pred[torch.arange(n),y_true])
    # Average over all samples
    loss = torch.mean(log_likelihood)

    # Caculate the gradient 
    grad = softmax
    softmax[torch.arange(n), y_true] -= 1
    grad /= n

    return loss, grad

def Softmax(z):
    exps = torch.exp(z)
    p = exps / torch.sum(exps, axis=1, keepdim=True)

    return p


# Training parameters
learning_rate = 5e-1  # step size for gradient descent
epochs = 10  # how many times to iterate through the intire training set


# Define list to store loss of each iteration
train_losses = []
train_accs = []

for epoch in range(epochs):
    # Training loop
    for i, (x_batch, y_batch) in enumerate(train_loader):
        # Flatten input to 1D tensor
        x_batch = x_batch.flatten(start_dim=1)
        # Convert labels to one-hot encoding
        y_batch = nn.functional.one_hot(y_batch, num_classes=10)

        # Perform forward pass
        y_pred = net.forward(x_batch)

        loss, grad = CrossEntropyLoss(y_batch, y_pred)
        train_losses.append(loss)
        
        net.backward(grad)
        net.optimizer_step(learning_rate)

        # Calculate accuracy of prediction
        correct = torch.argmax(y_pred, axis=1) == torch.argmax(y_batch, axis=1)
        train_accs.append(torch.sum(correct)/len(y_pred))

        # Print progress
        if i % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, i * len(x_batch), len(train_loader.dataset),
                100. * i / len(train_loader), loss))



