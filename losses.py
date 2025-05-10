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


class TVLoss(torch.nn.Module):
        def __init__(self, falseP_weight=0.5, eps=1e-8):
            """
            :param falseP_weight: number from 0-1 , weight higher to penalize false positives more
            :param eps: keeps denominator finite
            """
            super().__init__()
            self.eps = eps
            self.alpha = falseP_weight
            self.beta = 1 - falseP_weight

        def forward(self, pred, lab):
            trueP = (pred * lab).sum(axis=-1).sum(axis=-1)
            falseP = (pred * (1-lab)).sum(axis=-1).sum(axis=-1)
            falseN = ((1 - pred) * lab).sum(axis=-1).sum(axis=-1)
            tv_idx = trueP / (trueP + self.alpha*falseP + self.beta*falseN + self.eps)
            loss = 1-tv_idx
            loss = loss.mean()
            return loss