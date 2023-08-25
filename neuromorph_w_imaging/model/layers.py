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

class patchPredictor(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super(patchPredictor, self).__init__()
        # define the CNN patch encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=hidden_dim, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=hidden_dim, out_channels=out_dim, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.ReLU(inplace=True)
        )

        self.pooling = nn.AdaptiveAvgPool3d(1)

    def forward(self, im_patches):
        # encode the patch
        out = self.encoder(torch.permute(im_patches, (1,0,2,3,4)))
        out = self.pooling(out)
        return torch.squeeze(out)
        
class context_net(nn.Module):
    # expects input size of (batch_size, 1, 5, 13, 13)
    def __init__(self, hidden_dim, out_dim):
        super(context_net, self).__init__()
        # define the CNN patch encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=hidden_dim, kernel_size=(3,3,3), dilation=(1,2,2)),   # i (5, 13, 13) o (3, 9, 9)
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(3,3,3), dilation=(1,2,2)),   # i (3, 9, 9) o (1, 5, 5)
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1,3,3)),                     # i (1, 5, 5) o (1, 3, 3)
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=hidden_dim, out_channels=out_dim, kernel_size=(1,3,3)),                     # i (1, 3, 3) o (1, 1, 1)
        )

    def forward(self, im_patches):
        # encode the patch
        out = self.encoder(torch.permute(im_patches, (1,0,2,3,4)))
        return torch.squeeze(out)

# class context_net_norm(nn.Module):                                # does worse than context_net
#     # expects input size of (batch_size, 1, 5, 13, 13)
#     def __init__(self, hidden_dim, out_dim):
#         super(context_net_norm, self).__init__()
#         # define the CNN patch encoder
#         self.encoder = nn.Sequential(
#             nn.Conv3d(in_channels=1, out_channels=hidden_dim, kernel_size=(3,3,3), dilation=(1,2,2)),   # i (5, 13, 13) o (3, 9, 9)
#             nn.InstanceNorm3d(num_features=hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(3,3,3), dilation=(1,2,2)),   # i (3, 9, 9) o (1, 5, 5)
#             nn.InstanceNorm3d(num_features=hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1,3,3)),                     # i (1, 5, 5) o (1, 3, 3)
#             nn.InstanceNorm3d(num_features=hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(in_channels=hidden_dim, out_channels=out_dim, kernel_size=(1,3,3)),                     # i (1, 3, 3) o (1, 1, 1)
#         )

#     def forward(self, im_patches):
#         # encode the patch
#         out = self.encoder(torch.permute(im_patches, (1,0,2,3,4)))
#         return torch.squeeze(out)

class more_context_net(nn.Module):
    # expects input size of (batch_size, 1, 7, 19, 19)
    def __init__(self, hidden_dim, out_dim):
        super(more_context_net, self).__init__()
        # define the CNN patch encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=hidden_dim, kernel_size=(3,3,3), dilation=(1,3,3)),           # i (7, 19, 19) o (5, 13, 13)
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(3,3,3), dilation=(1,2,2)),           # i (5, 13, 13) o (3, 9, 9)
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(3,3,3), dilation=(1,2,2)),  # i (3, 9, 9) o (1, 5, 5)
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1,3,3)),                    # i (1, 5, 5) o (1, 3, 3)
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=hidden_dim, out_channels=out_dim, kernel_size=(1,3,3)),                       # i (1, 3, 3) o (1, 1, 1)
        )

    def forward(self, im_patches):
        # encode the patch
        out = self.encoder(torch.permute(im_patches, (1,0,2,3,4)))
        return torch.squeeze(out)