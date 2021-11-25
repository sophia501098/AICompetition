import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class myLoss(nn.Module):
    def __init__(self, hypara):
        super(myLoss, self).__init__()

    def forward(self, predict,label):
        loss_whole = 0
        return loss_whole