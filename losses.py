import torch

class diceLoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, lab):
        numer = (pred*lab).sum(axis=-1).sum(axis=-1)
        denom= pred.sum(axis=-1).sum(axis=-1) + lab.sum(axis=-1).sum(axis=-1) + self.eps
        dloss = 1-2*numer/denom
        dloss = dloss.mean()
        return dloss