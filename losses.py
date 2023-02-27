from cmath import exp, log
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TwoPlus1TupletLoss(nn.Module):
    """
    (N+1)Tuplet loss: when N=2, it is equivalent to triplet loss.
    Sohn, K., 2016. Improved deep metric learning with multi-class n-pair loss objective. 
    Advances in neural information processing systems, 29.
    """
    def __init__(self):
        super(TwoPlus1TupletLoss, self).__init__()

    def forward(self, anchor, positive, negative): 
        # when euclidean distance is used, this formula should be modified!
        loss_tuplet= torch.log(1+ torch.exp(torch.cosine_similarity(anchor,negative)-torch.cosine_similarity(anchor, positive)))
        return torch.mean(loss_tuplet)


class CombinedLoss(nn.Module):
    """
    Metric loss and and Classification Loss is combined using a lamda parameter to adjust the weight of the metric loss
    Binary mask ensures that each sample is counted at most once during CrossEntropyLoss calculation.  
    """
    def __init__(self, nbr_shot, lamda= None):
        super(CombinedLoss, self).__init__()
        self.lamda= lamda
        self.metric_loss = TwoPlus1TupletLoss()
        self.cls_loss= nn.CrossEntropyLoss(reduction='mean')
        if nbr_shot == 1:
            self.mask = np.array([True, False, False, False, False, False, False, False, True, True, True, True])
        else: # shots==5
            self.mask = np.array(
            [True, False, False, False, False, False, False, False, False, False, False, False,	False, False, False, False, False, False, False, False, 
            True, False, False, False, False, True, False, False, False, False, True, False, False, False, False, True, False, False, False, False,
            True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True ,True, True])

    def forward(self, anchor, positive, negative, cls_probas, target): # anchor, positive, negative shape [batch_size, emb_size]; cls_probas shape [batch_size*3, 5(num_cls)]; target shape [batch_size*3]
        loss_metric= self.metric_loss(anchor, positive, negative) 
        loss_cls = self.cls_loss(cls_probas[self.mask], target[self.mask])

        if self.lamda is not None:
            return self.lamda*loss_metric + loss_cls
        return loss_metric + loss_cls


