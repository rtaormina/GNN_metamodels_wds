# Libraries
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric
import torch_geometric.nn.models
from torch import Tensor
from torch.nn import Linear, Tanh, Sequential, LayerNorm, ReLU, Sigmoid, Dropout, PReLU, LeakyReLU
from torch_geometric.nn import GCNConv, ChebConv, MessagePassing
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_geometric.nn.inits import reset, glorot, uniform
import torch.nn.modules.activation as activations
import torch_geometric.nn.norm as norms
import torch_geometric.nn.dense.linear as pyg_linear
from typing import Union, Optional, Tuple, List



class NNConvEmbed(MessagePassing):
    def __init__(self, x_num, ea_num, emb_channels, aggr, dropout_rate=0.1):
        super(NNConvEmbed, self).__init__(aggr=aggr)

        self.x_num = x_num
        self.ea_num = ea_num
        self.emb_channels = emb_channels
        self.nn = Sequential(Linear(2 * x_num + ea_num, emb_channels), Dropout(p=dropout_rate), ReLU())
        self.aggr = aggr
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        return out

    def message(self, x_i, x_j, edge_attr):
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.nn(z)

    def __repr__(self):
        return '{}(aggr="{}", nn={})'.format(self.__class__.__name__, self.aggr, self.nn)


class GNN_ChebConv(nn.Module):
    def __init__(self, hid_channels, edge_features, node_features, edge_channels=32, dropout_rate=0, CC_K=2,
                 emb_aggr='max', depth=2, normalize=True):
        super(GNN_ChebConv, self).__init__()
        self.hid_channels = hid_channels
        self.dropout = dropout_rate
        self.normalize = normalize

        # embedding of node/edge features with NN
        self.embedding = NNConvEmbed(node_features, edge_features, edge_channels, aggr=emb_aggr)

        # CB convolutions (with normalization)
        self.convs = nn.ModuleList()
        for i in range(depth):
            # if normalize == True:
            # self.convs.append(LayerNorm(edge_channels, elementwise_affine=True))
            if i == 0:
                self.convs.append(ChebConv(edge_channels, hid_channels, CC_K, normalization='sym'))
            else:
                self.convs.append(ChebConv(hid_channels, hid_channels, CC_K, normalization='sym'))

        # output layer (so far only a 1 layer MLP, make more?)
        if depth == 0:
            self.lin = Linear(edge_channels, 1)
        else:
            self.lin = Linear(hid_channels, 1)
        # self.out = torch.nn.Sigmoid()       

    def forward(self, data):

        # retrieve model device (for LayerNorm to work)
        device = next(self.parameters()).device
        # data = data.to(device)

        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # 1. Pre-process data (nodes and edges) with MLP
        x = self.embedding(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # 2. Do convolutions
        for i in range(len(self.convs)):
            x = self.convs[i](x=x, edge_index=edge_index)
            if self.normalize:
                x = nn.LayerNorm(self.hid_channels, eps=1e-5, device=device)(x)
            x = F.dropout(x, self.dropout, training=self.training)
            x = nn.ReLU()(x)

        # 3. Output
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        # x = self.out(x)

        # Mask over storage nodes (which have pressure=0)
        x = (x[:, 0] * (1 - data.x[:, 2])).unsqueeze(-1)

        return x



class ModuleList(torch.nn.ModuleList):

    def forward(self, x):
        for l in self:
            x = l(x)
        return x

    def reset_parameters(self):
        for l in self:
            if hasattr(l, 'reset_parameters'):
                l.reset_parameters()
