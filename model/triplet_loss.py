import torch.nn as tnn


class TripletMarginLoss:
    def __init__(self):
        self.margin = 1

    def loss(self,dataset):

        triplet_loss = tnn.TripletMarginLoss(margin=self.margin)

        output = triplet_loss(anchor, positive, negative)

        return triplet_loss
