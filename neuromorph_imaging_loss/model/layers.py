from torch import nn
from torch_geometric.nn import EdgeConv
from model.layers_onet import ResnetBlockFC
from utils.base_tools import *


def MLP(channels):
    return nn.Sequential(*[
        nn.Sequential(nn.Linear(channels[i - 1], channels[i]),
                            nn.ReLU(), nn.BatchNorm1d(channels[i]))
        for i in range(1, len(channels))
    ])


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class ResnetECPos(nn.Module):
    def __init__(self, c_dim=128, dim=3, hidden_dim=128):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.block_0 = EdgeConv(ResnetBlockFC(4*hidden_dim, hidden_dim))
        self.block_1 = EdgeConv(ResnetBlockFC(4*hidden_dim+2*dim, hidden_dim))
        self.block_2 = EdgeConv(ResnetBlockFC(4*hidden_dim+2*dim, hidden_dim))
        self.block_3 = EdgeConv(ResnetBlockFC(4*hidden_dim+2*dim, hidden_dim))
        self.block_4 = EdgeConv(ResnetBlockFC(4*hidden_dim+2*dim, hidden_dim))
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p, edge_index):
        net = self.fc_pos(p)
        net = self.block_0(net, edge_index)

        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled, p], dim=1)

        net = self.block_1(net, edge_index)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled, p], dim=1)

        net = self.block_2(net, edge_index)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled, p], dim=1)

        net = self.block_3(net, edge_index)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled, p], dim=1)

        net = self.block_4(net, edge_index)

        c = self.fc_c(self.actvn(net))

        return c
