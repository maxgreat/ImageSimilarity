import torch
import torch.nn as nn

class EuclideanLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin=margin

    def forward(self,x,y,p):
        s = 0
        pdist = nn.PairwiseDistance()
        dists = pdist(x,y)
        for i, eq in enumerate(p):
            if eq == 1:
                s += dists[i]
            else:
                s+= max(0, self.margin - dists[i])
        return s

class CosineLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin=margin

    def forward(self,x,y,p):
        s = 0
        dists = f.cosine_similarity(x,y)
        for i, eq in enumerate(p):
            if eq == 1:
                s += 1-dists[i] #equal means == 1
            else:
                s+= max(0, dists[i] - self.margin) #similarity must be smaller than margin
        return s

class EuclideanTriple(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin=margin

    def forward(self,x,y,z):
        s = 0
        pdist = nn.PairwiseDistance()
        distsP = pdist(x,y)
        distsN = pdist(x,z)
        for i,d in enumerate(distsP):
            s = max(0, d + self.margin - distsN[i])
        return s
