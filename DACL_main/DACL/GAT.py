
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, alpha = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size = (in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data)
        self.a = nn.Parameter(torch.empty(size = (2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data)
        self.W_1 = nn.Parameter(torch.randn(in_features, out_features))
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        W_h = torch.matmul(h, self.W)
        W_adj = torch.mm(adj, self.W)
        a_input = torch.cat((W_h.repeat(W_adj.shape[0], 1), W_adj), dim = 1)
        attention = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(-1)

        attention = F.softmax(attention, dim = -1)
        W_adj_transform = torch.mm(adj, self.W_1)
        h = torch.matmul(attention, W_adj_transform)

        attention1 = attention.tolist()
        mean_of_list = sum(attention1) / len(attention1)
        attention2 = []
        for i in range(len(attention1)):
            if attention1[i] > mean_of_list:
                attention2.append(1)
            else:
                attention2.append(0)

        if len(attention2) == 1:
            attention2 = []
        if len(attention2) == 0:
            attention2 = []
        if all(element == 0 for element in attention2):
            attention2 = []

        return h, attention2
