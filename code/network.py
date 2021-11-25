import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class myNetwork(nn.Module):
    def __init__(self, hypara):
        super(myNetwork, self).__init__()

    def forward(self,input):
        return 1;
