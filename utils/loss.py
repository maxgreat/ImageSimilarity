import torch
import torch.nn as nn
import torch.nn.functional as f

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
                s += max(0, 1-self.margin-dists[i]) #equal means == 1
            else:
                s+= max(0, dists[i] - (1-self.margin)) #similarity must be smaller than margin
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
        
        
        
class ContractiveLoss2(nn.Module):
    def __init__(self, margin=2.0):
        super(ContractiveLoss2, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = f.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, imgs, caps):
        scores = torch.mm(imgs, caps.t())
        diag = scores.diag()

        cost_s = torch.clamp((self.margin - diag).expand_as(scores) + scores, min=0)

        # compare every diagonal score to scores in its row (i.e, all
        # contrastive sentences for each image)
        cost_im = torch.clamp((self.margin - diag.view(-1, 1)).expand_as(scores) + scores, min=0)
        # clear diagonals
        diag_s = torch.diag(cost_s.diag())
        diag_im = torch.diag(cost_im.diag())

        cost_s = cost_s - diag_s
        cost_im = cost_im - diag_im

        return cost_s.sum() + cost_im.sum()


class HardNegativeContrastiveLoss(nn.Module):
    def __init__(self, nmax=1, margin=0.2):
        super(HardNegativeContrastiveLoss, self).__init__()
        self.margin = margin
        self.nmax = nmax

    def forward(self, imgs, caps):
        scores = torch.mm(imgs, caps.t())
        diag = scores.diag()

        # Reducing the score on diagonal so there are not selected as hard negative
        scores = (scores - 2 * torch.diag(scores.diag()))

        sorted_cap, _ = torch.sort(scores, 0, descending=True)
        sorted_img, _ = torch.sort(scores, 1, descending=True)

        # Selecting the nmax hardest negative examples
        max_c = sorted_cap[:self.nmax, :]
        max_i = sorted_img[:, :self.nmax]

        # Margin based loss with hard negative instead of random negative
        neg_cap = torch.sum(torch.clamp(max_c + (self.margin - diag).view(1, -1).expand_as(max_c), min=0))
        neg_img = torch.sum(torch.clamp(max_i + (self.margin - diag).view(-1, 1).expand_as(max_i), min=0))

        loss = neg_cap + neg_img

        return loss
