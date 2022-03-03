from cmath import exp, log
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CombinedLoss(nn.Module):
    """
    Triplet and Classification Loss
    """
    def __init__(self, margin):
        super(CombinedLoss, self).__init__()
        self.margin= margin
        self.triplet_loss = nn.TripletMarginLoss(self.margin, p=2) #reduction='mean' by default
        self.cls_loss= nn.CrossEntropyLoss(reduction='mean')
        self.mask = np.array([True, False, False, False, False, False, False, False, True, True, True, True])

    def forward(self, anchor, positive, negative, cls_probas, target): # anchor, positive, negative shape [batch_size, emb_size]; cls_probas shape [batch_size*3, 5(num_cls)]; target shape [batch_size*3]
        loss_triplet= self.triplet_loss(anchor, positive, negative) 
        loss_cls = self.cls_loss(cls_probas[self.mask], target[self.mask])

        return loss_triplet+loss_cls

    

class TripletNoMarginLoss(nn.Module):
    def __init__(self):
        super(TripletNoMarginLoss, self).__init__()

    def forward(self, anchor, positive, negative): # anchor, positive, negative shape [batch_size, emb_size]; cls_probas shape [batch_size*3, 5(num_cls)]; target shape [batch_size*3]
        loss_triplet= torch.log(1+ torch.exp(torch.cosine_similarity(anchor,negative)-torch.cosine_similarity(anchor, negative)))
        return torch.mean(loss_triplet)


class CombinedLoss2(nn.Module):
    """
    Triplet and Classification Loss
    """
    def __init__(self, lamda, nbr_shot):
        super(CombinedLoss2, self).__init__()
        self.lamda= lamda
        self.triplet_loss = TripletNoMarginLoss()
        self.cls_loss= nn.CrossEntropyLoss(reduction='mean')
        if nbr_shot == 1:
            self.mask = np.array([True, False, False, False, False, False, False, False, True, True, True, True])
        else: # shots==5
            self.mask = np.array(
            [True, False, False, False, False, False, False, False, False, False, False, False,	False, False, False, False, False, False, False, False, 
            True, False, False, False, False, True, False, False, False, False, True, False, False, False, False, True, False, False, False, False,
            True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True ,True, True])
        
    def forward(self, anchor, positive, negative, cls_probas, target): # anchor, positive, negative shape [batch_size, emb_size]; cls_probas shape [batch_size*3, 5(num_cls)]; target shape [batch_size*3]
        loss_triplet= self.triplet_loss(anchor, positive, negative) 
        loss_cls = self.cls_loss(cls_probas[self.mask], target[self.mask])

        return self.lamda*loss_triplet+loss_cls