import torch 
from torch import nn
import torch.nn.functional as F 



class FocalBCELoss(nn.Module):
    '''
    Multi-class Focal loss implementation
    '''
    def __init__(self, gamma=2, alpha = 0.25, reduction="none"):
        super(FocalBCELoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, inputs, targets, weight):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, weight=weight)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

#         if self.alpha >= 0:
#             alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
#             loss = alpha_t * loss

        # Check reduction option and return loss accordingly
    
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return loss

class FocalBCELoss1(nn.Module):
    '''
    Multi-class Focal loss implementation
    '''
    def __init__(self, gamma=2):
        super(FocalBCELoss1, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets, weight):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = torch.sigmoid(inputs)
#         pt = torch.exp(logpt)
        logpt = (1-logpt)**self.gamma 
        print('log',logpt.shape)
        print('targets', targets.shape)
        print('weight', weight.shape)
        loss = F.binary_cross_entropy(logpt, targets, weight=weight)
        return loss
    
    
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
    
    def forward(self, preds, target, alpha=0.7, beta = 0.3, epsilon=1e-6):
        preds = torch.sigmoid(preds) 

        #flatten label and preds tensors
        preds = preds.reshape(-1)
        target = target.reshape(-1)

        #True Positives, False Positives & False Negatives 
        TP = (preds * target).sum()
        FP = ((1-target) * preds).sum()
        FN = (target * (1-preds)).sum()
        Tversky = (TP + epsilon)/(TP + alpha*FP + beta*FN + epsilon)

        return 1 - Tversky 

class FocalTverskyLoss(nn.Module):
    def __init__(self):
        super(FocalTverskyLoss, self).__init__()
    
    def forward(self, preds, target, alpha=0.65, beta = 0.35, epsilon=1e-6, gamma=3):
        preds = torch.sigmoid(preds) 

        #flatten label and preds tensors
        preds = preds.reshape(-1)
        target = target.reshape(-1)

        #True Positives, False Positives & False Negatives 
        TP = (preds * target).sum()
        FP = ((1-target) * preds).sum()
        FN = (target * (1-preds)).sum()
        Tversky = (TP + epsilon)/(TP + alpha*FP + beta*FN + epsilon)
        FocalTversky = (1 - Tversky)**gamma

        return FocalTversky