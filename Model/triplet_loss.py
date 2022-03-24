import torch.nn as tnn


class TripletMarginLoss:
    def __init__(self):
        self.margin = 1

    def loss(self):
        triplet_loss = tnn.TripletMarginLoss(margin=self.margin)

        return triplet_loss
