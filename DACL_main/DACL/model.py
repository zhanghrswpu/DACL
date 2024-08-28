import torch
import torch.nn as nn
from GAT import GraphAttentionLayer

class model(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.GAT_neighbor = GraphAttentionLayer(embed_size, embed_size)
        self.GAT_item = GraphAttentionLayer(embed_size, embed_size)
        self.relation_neighbor = nn.Parameter(torch.rand(1))
        self.relation_item = nn.Parameter(torch.rand(1))
        self.relation_self = nn.Parameter(torch.rand(1))


    def predict(self, user_embedding, item_embedding):
        return torch.matmul(user_embedding, item_embedding.t())


    def forward(self, feature_self, feature_neighbor, feature_item):
        if type(feature_item) == torch.Tensor:
            f_n,attention1= self.GAT_neighbor(feature_self, feature_neighbor)
            f_i,attention2 = self.GAT_item(feature_self, feature_item)
            m = nn.Softmax(dim=0)
            e_tensor = torch.stack([self.relation_neighbor, self.relation_item, self.relation_self])
            e_tensor = m(e_tensor)
            relation_neighbor, relation_item, relation_self = e_tensor
            user_embedding = relation_self * feature_self +relation_neighbor * f_n + relation_item *f_i
        else:
            f_n,attention1 = self.GAT_neighbor(feature_self, feature_neighbor)
            m = nn.Softmax(dim=0)
            e_tensor = torch.stack([self.relation_neighbor, self.relation_self])
            e_tensor = m(e_tensor)
            relation_neighbor, relation_self = e_tensor
            user_embedding = relation_self * feature_self + relation_neighbor * f_n

        return user_embedding, attention1

