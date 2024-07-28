import numpy as np
from itertools import product
import torch
import dgl

def loadGraphData(dataset, label, ratio=0.5, constant=0.01):
    hetero_graph = makeGraph(dataset, label, ratio, constant)
    return hetero_graph


def makeGraph(dataset, label, ratio, constant):
    n_row = dataset.shape[0]
    n_col = dataset.shape[1]
    row_ids = np.arange(n_row)
    col_ids = np.arange(n_col)
    n_classes = int(np.max(label)) + 1
    edge_indices = torch.from_numpy(np.array(list(map(list, list(product(row_ids, col_ids))))).T).long()
    edge_mask = torch.zeros(n_row * n_col, dtype=torch.bool).bernoulli(ratio)
    edge_mask[0], edge_mask[-1] = True, True  # 防止未采样第一个与最后一个元素
    # nonzero_indices = np.array(torch.nonzero(edge_mask).squeeze())
    # indices = np.random.choice(nonzero_indices, size=int(len(nonzero_indices)*0.1), replace=False)
    # train_edge_mask = edge_mask.clone()
    # train_edge_mask[indices] = False
    #
    # mask = torch.zeros(len(torch.nonzero(edge_mask).squeeze()), dtype=torch.bool).bernoulli(0.9)
    # mask[0], mask[-1] = True


    hetero_graph = dgl.heterograph({
        ('measurement', 'link', 'window'): (edge_indices[0][edge_mask], edge_indices[1][edge_mask]),
        ('window', 'linked-by', 'measurement'): (edge_indices[1][edge_mask], edge_indices[0][edge_mask])
    })

    hetero_graph.nodes['measurement'].data['feature'] = constant * torch.ones(n_row, n_col).float()
    hetero_graph.nodes['measurement'].data['label'] = torch.from_numpy(label).long()
    hetero_graph.nodes['measurement'].data['lack'] = edge_mask.reshape(n_row, n_col)
    hetero_graph.nodes['measurement'].data['edge'] = torch.from_numpy(dataset).float()
    # hetero_graph.nodes['measurement'].data['train-lack'] = train_edge_mask.reshape(n_row, n_col)
    # hetero_graph.edges['link'].data['mask'] = mask
    # hetero_graph.edges['linked-by'].data['mask'] = mask

    # pe = torch.zeros(n_col, n_col)
    # position = torch.arange(1, n_col+1).unsqueeze(1).float()
    # exp_term = torch.exp(torch.arange(0, n_col, 2).float() * -(torch.log(torch.tensor(10000.0)) / n_col))
    # print(exp_term)
    # pe[:, 0::2] = torch.sin(position * exp_term)  # take the odd (jump by 2)
    # pe[:, 1::2] = torch.cos(position * exp_term)  # take the even (jump by 2)

    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / n_col) for j in range(n_col)]
        if pos != 0 else np.zeros(n_col) for pos in range(n_col)])
    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # apply sin on 0th,2nd,4th...emb_dim
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # apply cos on 1st,3rd,5th...emb_dim

    hetero_graph.nodes['window'].data['feature'] = constant * torch.from_numpy(position_enc).float()
    #hetero_graph.nodes['window'].data['feature'] = constant * torch.eye(n_col).float()
    hetero_graph.edges['link'].data['label'] = torch.from_numpy(dataset.reshape(n_row * n_col, -1)[edge_mask]).float()
    hetero_graph.edges['linked-by'].data['label'] = torch.from_numpy(dataset.reshape(n_row * n_col, -1)[edge_mask]).float()

    # if is_train:
    #     hetero_graph.nodes['measurement'].data['train_mask'] = torch.zeros(n_row, dtype=torch.bool).bernoulli(0.9)
    # else:
    #     hetero_graph.nodes['measurement'].data['train_mask'] = torch.ones(n_row, dtype=torch.bool)
    hetero_graph.nodes['measurement'].data['train_mask'] = torch.ones(n_row, dtype=torch.bool)
    return hetero_graph


