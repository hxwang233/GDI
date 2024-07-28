import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from models.epred import EdgePredictor
from models.esage import ESAGEConv
from models.conv import ConvNet
from models.lstm import RNN

class Model(nn.Module):
    def __init__(self, n_in_feats, e_in_feats, n_hidden_feats, e_hidden_feats,
                n_out_feats, e_out_feats, n_class, n_col, n_layer,
                node_attention=True, edge_attention=False, classify_name = "cnn"):
        super().__init__()
        self.classify_name = classify_name
        self.n_layer = n_layer
        self.esage1 = ESAGEConv(n_in_feats=n_in_feats, e_in_feats=e_in_feats, n_out_feats=n_hidden_feats, e_out_feats=e_hidden_feats)
        self.esage2 = ESAGEConv(n_in_feats=n_hidden_feats, e_in_feats=e_hidden_feats, n_out_feats=n_out_feats, e_out_feats=e_out_feats)
        self.epred = EdgePredictor(row_in_feats=n_out_feats, col_in_feats=n_hidden_feats, out_feats=e_out_feats)
        if self.classify_name == "cnn":
            self.classify = ConvNet(height=e_out_feats, width=n_in_feats, in_channels=1, node_feats=n_out_feats, n_class=n_class, channels=[64, 128, 64])
        elif self.classify_name == "lstm":
            self.classify = RNN(e_out_feats, e_hidden_feats, n_class, bidirectional=True)
        elif self.classify_name == "mlp":
            self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(e_out_feats*n_in_feats, n_class),
            nn.LogSoftmax(dim=1)
            )
        if edge_attention:
            self.edge_pe = nn.Embedding(n_col, e_out_feats)
            self.edge_pe.weight.data = self.position_encoding_init(n_col, e_out_feats)
            self.e_linear_q = nn.Linear(e_out_feats, e_out_feats, bias=False)
            self.e_linear_k = nn.Linear(e_out_feats, e_out_feats, bias=False)
            self.e_linear_v = nn.Linear(e_out_feats, e_out_feats, bias=False)
            self.edge_norm_fact = 1 / np.sqrt(e_out_feats)
            self.edge_refine = nn.Linear(e_out_feats, e_out_feats)
            self.edge_layer_norm = nn.LayerNorm([n_col, e_out_feats])
            self.edge_batch_norm = nn.BatchNorm1d(n_col)
            init.xavier_normal_(self.edge_refine.weight)
            init.xavier_normal_(self.e_linear_q.weight)
            init.xavier_normal_(self.e_linear_k.weight)
            init.xavier_normal_(self.e_linear_v.weight)
        if node_attention:
            self.node_pe = nn.Embedding(n_col, n_hidden_feats)
            self.node_pe.weight.data = self.position_encoding_init(n_col, n_hidden_feats)
            self.n_linear_q = nn.Linear(n_hidden_feats, n_hidden_feats, bias=False)
            self.n_linear_k = nn.Linear(n_hidden_feats, n_hidden_feats, bias=False)
            self.n_linear_v = nn.Linear(n_hidden_feats, n_hidden_feats, bias=False)
            self.node_norm_fact = 1 / np.sqrt(n_hidden_feats)
            self.node_refine = nn.Linear(n_hidden_feats, n_hidden_feats)
            self.node_layer_norm = nn.LayerNorm([n_col, n_hidden_feats])
            self.node_batch_norm = nn.BatchNorm1d(n_hidden_feats)
            init.xavier_normal_(self.node_refine.weight)
            init.xavier_normal_(self.n_linear_q.weight)
            init.xavier_normal_(self.n_linear_k.weight)
            init.xavier_normal_(self.n_linear_v.weight)
        self.node_attention = node_attention
        self.edge_attention = edge_attention
        self.activation = torch.nn.LeakyReLU()
        return

    def forward(self, blocks, x, e, real, mask):
        x, e = self.esage1(blocks[0], x, e)
        new_col = x[blocks[0].dsttypes[1]]  # col key
        # layer 2
        x, e = self.esage2(blocks[1], x, e)
        new_row = x[blocks[1].dsttypes[0]]  # row key
        # pred lack edge
        if self.node_attention:
            node_pos = torch.arange(mask.shape[1]).to(torch.device('cuda'))
            new_col  = new_col + self.node_pe(node_pos)
            q = self.n_linear_q(new_col)  # n, dim_k
            k = self.n_linear_k(new_col)  # n, dim_k
            v = self.n_linear_v(new_col)  # n, dim_v
            dist = torch.mm(q, k.transpose(0, 1)) * self.node_norm_fact  # n, n
            dist = torch.softmax(dist, dim=-1)  # n, n
            new_col = torch.mm(dist, v)
            new_col = new_col + torch.mm(dist, v)
            new_col = torch.sigmoid(self.node_refine(new_col))
            #new_col = torch.sigmoid(self.node_batch_norm(new_col))
            #new_col = torch.sigmoid(self.node_refine(new_col))
        e = self.epred(new_row, new_col, mask=mask)
        # aligning
        imput_mask = mask.flatten()
        imput_e = torch.zeros(imput_mask.shape[0], e.shape[1])
        if torch.cuda.is_available():
            imput_e = imput_e.to(torch.device('cuda'))
        if torch.cuda.is_available():
            imput_e = imput_e.to(torch.device('cuda'))
        imput_e[imput_mask, :] = real
        imput_e[~imput_mask, :] = e
        # attention refine
        imput_e = torch.reshape(imput_e, (new_row.shape[0], -1, imput_e.shape[-1]))
        if self.edge_attention:
            edge_pos = torch.arange(mask.shape[1]).repeat(mask.shape[0], 1).to(torch.device('cuda'))
            imput_e = imput_e + self.edge_pe(edge_pos)
            q = self.e_linear_q(imput_e)  # batch, n, dim_k
            k = self.e_linear_k(imput_e)  # batch, n, dim_k
            v = self.e_linear_v(imput_e)  # batch, n, dim_v
            dist = torch.bmm(q, k.transpose(1, 2)) * self.edge_norm_fact  # batch, n, n
            dist = torch.softmax(dist, dim=-1)  # batch, n, n
            imput_e = imput_e + torch.bmm(dist, v)
            #imput_e = torch.sigmoid(self.edge_layer_norm(imput_e))
            #imput_e = self.edge_batch_norm(imput_e)
            imput_e = torch.sigmoid(self.edge_refine(imput_e))
        # classify
        if self.classify_name == "cnn":
            res = self.classify(imput_e, new_row)
        else:
            res = self.classify(imput_e)
        return res, e

    def position_encoding_init(self, n_position, emb_dim):
        ''' Init the sinusoid position encoding table '''
        # keep dim 0 for padding token position encoding zero vector
        position_enc = np.array([
            [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
            if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])
        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # apply sin on 0th,2nd,4th...emb_dim
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # apply cos on 1st,3rd,5th...emb_dim
        return torch.from_numpy(position_enc).type(torch.FloatTensor)