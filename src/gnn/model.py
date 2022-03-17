# created by Wu Bolun
# 2020.11.9
# bowenwu@sjtu.edu.cn

import sys

import torch
import torch.nn.functional as F

from torch import nn
from torch.nn import Conv1d, MaxPool1d, Linear, Sequential, Dropout, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GCNConv, global_sort_pool, GINConv, global_mean_pool, JumpingKnowledge
from torch_geometric.utils import remove_self_loops


class DGCNN(nn.Module):
    ''' DGCNN '''
    def __init__(self, num_features, num_classes):
        super(DGCNN, self).__init__()

        self.k = 64

        # dgcnn
        self.conv1 = GCNConv(num_features, 128)
        self.conv2 = GCNConv(128, 64)
        self.conv3 = GCNConv(64, 32)
        self.conv4 = GCNConv(32, 32)
        self.conv5 = Conv1d(1, 16, 256, 256)
        self.conv6 = Conv1d(16, 32, 5, 1)
        self.pool = MaxPool1d(2, 2)
        self.classifier_1 = Linear(896, 128)
        self.drop_out = Dropout(0.5)
        self.classifier_2 = Linear(128, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # dgcnn
        x_1 = torch.tanh(self.conv1(x, edge_index))
        x_2 = torch.tanh(self.conv2(x_1, edge_index))
        x_3 = torch.tanh(self.conv3(x_2, edge_index))
        x_4 = torch.tanh(self.conv4(x_3, edge_index))
        x = torch.cat([x_1, x_2, x_3, x_4], dim=-1)
        x = global_sort_pool(x, batch, k=self.k)
        x = x.view(x.size(0), 1, x.size(-1))
        x = self.relu(self.conv5(x))
        x = self.pool(x)
        x = self.relu(self.conv6(x))
        x = x.view(x.size(0), -1)
        out = self.relu(self.classifier_1(x))
        out = self.drop_out(out)
        classes = F.log_softmax(self.classifier_2(out), dim=-1)

        # After conv4: torch.Size([43071, 32])
        # After cat: torch.Size([43071, 256])
        # After sort pool: torch.Size([20, 16384])
        # After view: torch.Size([20, 1, 16384])
        # After conv1d 1: torch.Size([20, 16, 64])
        # After max pool : torch.Size([20, 16, 32])
        # After conv1d 2: torch.Size([20, 32, 28])
        # before linear: torch.Size([20, 896])
        return classes


class GIN0(torch.nn.Module):
    def __init__(self, num_features, num_layers, hidden, num_classes):
        super(GIN0, self).__init__()
        self.conv1 = GINConv(
            Sequential(
                Linear(num_features, hidden),
                ReLU(),
                Linear(hidden, hidden),
                ReLU(),
                BN(hidden),
            ), train_eps=False)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        ReLU(),
                        BN(hidden),
                    ), train_eps=False))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class GIN0WithJK(torch.nn.Module):
    def __init__(self, num_features, num_layers, hidden, num_classes, mode='cat'):
        super(GIN0WithJK, self).__init__()
        self.conv1 = GINConv(
            Sequential(
                Linear(num_features, hidden),
                ReLU(),
                Linear(hidden, hidden),
                ReLU(),
                BN(hidden),
            ), train_eps=False)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        ReLU(),
                        BN(hidden),
                    ), train_eps=False))
        self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            self.lin1 = Linear(num_layers * hidden, hidden)
        else:
            self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        # xs = [x]
        xs = [global_mean_pool(x, batch)]
        for conv in self.convs:
            x = conv(x, edge_index)
            # xs += [x]
            xs += [global_mean_pool(x, batch)]
        x = self.jump(xs)
        # x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

