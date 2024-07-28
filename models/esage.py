import psutil
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import dgl

class ESAGEConv(nn.Module):
    def __init__(self, n_in_feats, e_in_feats, n_out_feats, e_out_feats, message_activation=torch.tanh, update_activation=torch.tanh, activation=torch.tanh):
        super(ESAGEConv, self).__init__()

        self.message_lin = nn.Linear(n_in_feats + e_in_feats, n_out_feats)
        self.update_node_lin = nn.Linear(n_in_feats + n_out_feats, n_out_feats)
        self.update_edge_lin = nn.Linear(n_in_feats + n_out_feats + e_in_feats, e_out_feats)
        init.xavier_normal_(self.message_lin.weight)
        init.xavier_normal_(self.update_edge_lin.weight)
        init.xavier_normal_(self.update_node_lin.weight)
        self.message_activation = message_activation
        self.update_activation = update_activation
        self.activation = activation
        self.drop = nn.Dropout(p=0.2)
        return

    def message_func(self, edges):
        m_j = torch.cat([edges.src['h'], edges.data['h']], dim=1)
        m_j = self.message_lin(m_j)
        m_j = self.message_activation(m_j)
        return {'m': m_j}

    def update_edge(self, edges):
        m_j = torch.cat([edges.src['h'], edges.dst['neigh'], edges.data['h']], dim=1)
        m_j = self.update_edge_lin(m_j)
        m_j = self.update_activation(m_j)
        return {'e': m_j}

    def forward(self, graph, feat, edge):
        for k in list(edge.keys()):
            if graph.num_edges(k) != 0:
                edge[k] = edge[k][:graph.num_edges(k)]
            else:
                edge.pop(k)
        with graph.local_scope():
            graph.srcdata['h'] = feat
            graph.edata['h'] = edge
            edge_types = list(edge.keys())
            edges, dstnodes = list(), list()
            for i in edge_types:
                edges.append(i[1])
                dstnodes.append(i[2])
            func = {i:(self.message_func, dgl.function.mean('m', 'neigh')) for i in edges}
            graph.multi_update_all(func, "mean")
            for n in dstnodes:
                h_neigh = self.update_node_lin(self.drop(torch.cat([graph.dstdata['neigh'][n], graph.srcdata['h'][n][:graph.num_dst_nodes(n)]], dim=1)))
                h_neigh = self.activation(h_neigh)
                graph.dstdata['neigh'][n] = h_neigh
            for e in edges:
                graph.apply_edges(self.update_edge, etype = e)
            return graph.dstdata['neigh'], graph.edata['e']
        return