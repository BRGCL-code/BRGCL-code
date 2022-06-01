import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GNNLayer(Module):
    def __init__(self, in_features, out_features, PReLU=False):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.PReLu = PReLU
        if PReLU:
            self.act = nn.PReLU()

        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        if active:
            if self.PReLu:
                output = self.act(output)
            else:
                output = F.relu(output)
        return output

