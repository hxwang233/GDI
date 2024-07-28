import numpy as np
import torch
import dgl
from sklearn.metrics import f1_score, recall_score, precision_score
from models.gnn import Model
from utils.early_stopping import EarlyStopping
import psutil
import time

def modelTrain(train_graph, valid_graph, test_graph, n_layer, n_in_feats, e_in_feats, n_hidden_feats, e_hidden_feats, n_out_feats, e_out_feats, n_class,
            batch_size=256, epoch=500, lr=0.005, alpha=0.6, train_edge_ratio=0.05, n_col=10, save_path="./", model_name="best_network.pth"):
    if torch.cuda.is_available():
        train_graph, valid_graph, test_graph = train_graph.to(torch.device('cuda')), valid_graph.to(torch.device('cuda')), test_graph.to(torch.device('cuda'))
    train_loader, valid_loader, test_loader = createLoader(train_graph, batch_size, n_layer), createLoader(valid_graph, batch_size, n_layer), createLoader(test_graph, batch_size, n_layer)
    early_stopping = EarlyStopping(save_path=save_path, model_name=model_name)
    model, loss_func, opt = createModel(n_layer, n_in_feats, e_in_feats, n_hidden_feats, e_hidden_feats, n_out_feats, e_out_feats, n_class, n_col, lr=lr)
    for e in range(epoch):
        model.train()
        total_loss_list, classify_loss_list, imput_loss_list = list(), list(), list()
        flag = 0
        for step, (input_nodes, output_nodes, blocks) in enumerate(train_loader):
            if step == 0:
                flag = len(input_nodes['window'])
            if len(input_nodes['window']) != flag:
                continue
            edge_features = blocks[0].edata['label']
            input_features = blocks[0].srcdata['feature']
            not_imput_features = blocks[-1].edata['label'][('window', 'linked-by', 'measurement')]
            imput_mask = blocks[-1].srcdata['lack']['measurement'][:, blocks[0].dstdata['_ID']['window']]
            class_real = blocks[-1].dstdata['label']['measurement']
            edge_real = blocks[-1].srcdata['edge']['measurement'][~imput_mask]
            opt.zero_grad()
            class_res, edge_res = model(blocks, input_features, edge_features, not_imput_features, imput_mask)
            classify_loss = loss_func[0](class_res, class_real)
            edge_loss_indices = np.random.choice(edge_res.shape[0], int(edge_res.shape[0]*train_edge_ratio), replace=False)
            imput_loss = loss_func[1](edge_res[edge_loss_indices], edge_real[edge_loss_indices])
            total_loss = classify_loss + alpha * imput_loss
            total_loss.backward()
            opt.step()
            total_loss_list.append(total_loss.detach().cpu().numpy())
            classify_loss_list.append(classify_loss.detach().cpu().numpy())
            imput_loss_list.append(imput_loss.detach().cpu().numpy())
        print("---第"+str(e+1)+"个epoch---")
        print('total loss:', np.mean(total_loss_list))
        print('classify loss:', np.mean(classify_loss_list))
        print('imput loss:', np.mean(imput_loss_list))
        eval_loss, f1, _ = myEval(model, valid_loader, loss_func, alpha)
        early_stopping(eval_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    model.load_state_dict(torch.load(save_path+model_name))  # 获得 early stopping 时的模型参数
    print("======Testset Eval======")
    test_loss, f1, imput_metric = myEval(model, test_loader, loss_func, alpha)
    return f1, imput_metric

def createLoader(graph, batch_size, n_layer):
    train_node_mask = graph.nodes['measurement'].data['train_mask']
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(n_layer)
    if torch.cuda.is_available():
        train_nid_dict = {'measurement': torch.arange(len(train_node_mask), dtype=torch.long)[train_node_mask].to(torch.device('cuda'))}
        dataloader = dgl.dataloading.DataLoader(graph, train_nid_dict, sampler, batch_size=batch_size,
                                                shuffle=True, drop_last=False,
                                                device=torch.device('cuda'))
    else:
        train_nid_dict = {'measurement': torch.arange(len(train_node_mask), dtype=torch.long)[train_node_mask]}
        dataloader = dgl.dataloading.DataLoader(graph, train_nid_dict, sampler, batch_size=batch_size,
                                                shuffle=True, drop_last=False)

    return dataloader

def createModel(n_layer, n_in_feats, e_in_feats, n_hidden_feats, e_hidden_feats, n_out_feats, e_out_feats, n_class, n_col, lr):
    loss_func1 = torch.nn.NLLLoss()
    loss_func2 = torch.nn.MSELoss()
    model = Model(n_in_feats, e_in_feats, n_hidden_feats, e_hidden_feats, n_out_feats, e_out_feats, n_class, n_col, n_layer)
    if torch.cuda.is_available():
        model = model.cuda()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    return model, [loss_func1, loss_func2], opt

def myEval(model, test_loader, loss_func, alpha):
    model.eval()
    classify_res_list, classify_real_list, imput_res_list, imput_real_list = np.array([]), np.array([]), None, None
    eval_loss = 0
    flag = 0
    for step, (input_nodes, output_nodes, blocks) in enumerate(test_loader):
        if step == 0:
            flag = len(input_nodes['window'])
        if len(input_nodes['window']) != flag:
            continue
        edge_features = blocks[0].edata['label']
        input_features = blocks[0].srcdata['feature']
        not_imput_features = blocks[-1].edata['label'][('window', 'linked-by', 'measurement')]
        imput_mask = blocks[-1].srcdata['lack']['measurement']
        class_real = blocks[-1].dstdata['label']['measurement']
        edge_real = blocks[-1].srcdata['edge']['measurement'][~imput_mask]
        class_res, edge_res = model(blocks, input_features, edge_features, not_imput_features, imput_mask)
        classify_loss = loss_func[0](class_res, class_real)
        imput_loss = loss_func[1](edge_res, edge_real)
        total_loss = classify_loss + alpha * imput_loss
        eval_loss += total_loss.detach().cpu().numpy()
        # metric
        class_real, class_res, edge_real, edge_res = class_real.detach().cpu().numpy(), class_res.detach().cpu().numpy(), \
                                                    edge_real.detach().cpu().numpy(), edge_res.detach().cpu().numpy()
        class_res = np.argmax(class_res, axis=1)
        imput_res_list  = edge_res if step == 0 else np.concatenate((imput_res_list, edge_res))
        imput_real_list = edge_real if step == 0 else np.concatenate((imput_real_list, edge_real))
        classify_res_list  = class_res if step == 0 else np.concatenate((classify_res_list, class_res))
        classify_real_list = class_real if step == 0 else np.concatenate((classify_real_list, class_real))
    mae = np.sum(np.fabs(imput_real_list - imput_res_list)) / (imput_real_list.shape[0] * imput_real_list.shape[1])     # imput_real_list.shape[0]
    mse = np.sum(np.square(imput_real_list - imput_res_list)) / (imput_real_list.shape[0] * imput_real_list.shape[1])   # imput_real_list.shape[0]
    nmae = np.sum(np.fabs(imput_real_list - imput_res_list)) / np.sum(np.fabs(imput_real_list))
    nmse = np.sum(np.square(imput_real_list - imput_res_list)) / np.sum(np.square(imput_real_list))
    f1_micro = f1_score(classify_real_list, classify_res_list, average="micro")
    f1_macro = f1_score(classify_real_list, classify_res_list, average="macro")
    recall = recall_score(classify_real_list, classify_res_list, average="macro")
    precision = precision_score(classify_real_list, classify_res_list, average="macro")
    print('--Eval--')
    print('f1_micro:', f1_micro, 'f1_macro:', f1_macro, 'recall:', recall, 'precision:', precision)
    print('MAE:', mae, 'NMAE:', nmae)
    print('MSE:', mse, 'RMSE:', np.sqrt(mse), 'NMSE:', nmse, 'NRMSE:', np.sqrt(nmse))
    return eval_loss, [f1_micro, f1_macro, recall, precision], [mae, np.sqrt(mse), nmae, np.sqrt(nmse)]


