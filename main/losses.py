import torch 
from torch import nn
import torch.nn.functional as F 


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)  
        #print(inputs)
        #inputs = (inputs>0.5).float()
        #inputs.requires_grad=True
        #print(inputs)
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class TverskyLoss(nn.Module):
    def __init__(self):
        super(TverskyLoss, self).__init__()
    
    def forward(self, preds, target, alpha=0.7, beta = 0.3, epsilon=1e-6, gamma=3):
        preds = torch.sigmoid(preds) 

        #flatten label and preds tensors
        preds = preds.reshape(-1)
        target = target.reshape(-1)

        #True Positives, False Positives & False Negatives 
        TP = (preds * target).sum()
        FP = ((1-target) * preds).sum()
        FN = (target * (1-preds)).sum()
        Tversky = (TP + epsilon)/(TP + alpha*FP + beta*FN + epsilon)
    #     FocalTversky = (1 - Tversky)**gamma

        return 1 - Tversky 