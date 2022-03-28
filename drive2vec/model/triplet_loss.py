import torch
import torch.nn as nn
import data

class TripletMarginLoss:
    def __init__(self):
        self.margin = 1

    def loss(self):
        TP = data.TripletDataset()

        triplet_data = TP.load_triplet_data()

        triplet_loss = TripletLoss(margin=self.margin)
        tensor_A = torch.tensor(triplet_data['A'].to_numpy)
        tensor_P = torch.tensor(triplet_data['P'].to_numpy)
        tensor_N = torch.tensor(triplet_data['N'].to_numpy)

        output = triplet_loss.forward(tensor_A,tensor_P,tensor_N)
        return output


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()