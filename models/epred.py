import torch
import torch.nn as nn
import torch.nn.init as init

class EdgePredictor(nn.Module):
    def __init__(self, row_in_feats, col_in_feats, out_feats, activation=torch.sigmoid):
        super().__init__()
        self.pred = nn.Linear(row_in_feats + col_in_feats, out_feats)
        self.drop = nn.Dropout(p=0.2)
        #self.pred = nn.Linear(in_feats + edge_feats, out_feats)
        self.activation = activation
        return

    def forward(self, node1, node2, mask):
        n1, n2 = mask.shape[0], mask.shape[1]
        mask = mask.flatten()
        node1 = node1.repeat_interleave(n2, 0)
        node2 = node2.repeat(n1, 1)
        e = torch.concat([node1, node2], dim=1)[~mask,:]
        e = self.activation(self.pred(self.drop(e)))
        return e