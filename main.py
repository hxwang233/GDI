import numpy as np

from load_config import get_yaml_data
from train_y import modelTrain
from utils.graph_data import loadGraphData
from sklearn.model_selection import train_test_split
import os

if __name__ == '__main__':

    config = get_yaml_data("config/config.yaml")

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config["experiment"]["gpu"])

    data = np.load(config["data_path"]["dataset_path"])
    label = np.load(config["data_path"]["label_path"])
    trainset, testset, trainlabel, testlabel = train_test_split(data, label, test_size=int(np.floor(data.shape[0] * 0.2)), stratify=label)
    trainset, validset, trainlabel, validlabel = train_test_split(trainset, trainlabel, test_size=int(trainset.shape[0] * 0.1), stratify=trainlabel)

    train_edge_ratio = config["experiment"]["train_edge_ratio"]
    repeat_time = config["experiment"]["repeat_time"]
    ratio_list = config["experiment"]["ratio_list"]
    batch_size = config["experiment"]["batch_size"]
    epoch = config["experiment"]["epoch"]
    lr = config["experiment"]["lr"]

    n_factor = config["model"]["n_factor"]
    n_layer = config["model"]["n_layer"]
    alpha = config["model"]["alpha"]

    n_col, n_e = data.shape[1], data.shape[2]
    labels = sorted(list(map(int, set(trainlabel))))
    total_f1_micro_list, total_f1_macro_list, total_recall_list, total_precision_list, \
    total_mae_list, total_rmse_list, total_nmae_list, total_nrmse_list = [], [], [], [], [], [], [], []

    for ratio in ratio_list:
        f1_micro_list, f1_macro_list, recall_list, precision_list, mae_list, rmse_list, nmae_list, nrmse_list = [], [], [], [], [], [], [], []
        for i in range(repeat_time):
            train_graph = loadGraphData(dataset=trainset, label=trainlabel, ratio=ratio-train_edge_ratio)
            valid_graph = loadGraphData(dataset=validset, label=validlabel, ratio=ratio)
            test_graph = loadGraphData(dataset=testset, label=testlabel, ratio=ratio)
            f1, imput_metric = modelTrain(train_graph, valid_graph, test_graph, n_layer=n_layer, n_in_feats=n_col,
                                        e_in_feats=n_e, n_hidden_feats=n_factor*2, e_hidden_feats=n_factor*2,
                                        n_col=n_col, n_out_feats=n_factor, e_out_feats=n_e, alpha=alpha,
                                        n_class=len(labels), train_edge_ratio=train_edge_ratio,
                                        epoch=epoch, batch_size=batch_size, lr=lr)
            f1_micro_list.append(f1[0])
            f1_macro_list.append(f1[1])
            recall_list.append(f1[2])
            precision_list.append(f1[3])
            mae_list.append(imput_metric[0])
            rmse_list.append(imput_metric[1])
            nmae_list.append(imput_metric[2])
            nrmse_list.append(imput_metric[3])
        total_f1_micro_list.append(np.mean(f1_micro_list))
        total_f1_macro_list.append(np.mean(f1_macro_list))
        total_recall_list.append(np.mean(recall_list))
        total_precision_list.append(np.mean(precision_list))
        total_mae_list.append(np.mean(mae_list))
        total_rmse_list.append(np.mean(rmse_list))
        total_nmae_list.append(np.mean(nmae_list))
        total_nrmse_list.append(np.mean(nrmse_list))
    total_f1_micro_list = np.array(total_f1_micro_list).reshape(1, -1)
    total_f1_macro_list = np.array(total_f1_macro_list).reshape(1, -1)
    total_recall_list = np.array(total_recall_list).reshape(1, -1)
    total_precision_list = np.array(total_precision_list).reshape(1, -1)
    total_mae_list = np.array(total_mae_list).reshape(1, -1)
    total_rmse_list = np.array(total_rmse_list).reshape(1, -1)
    total_nmae_list = np.array(total_nmae_list).reshape(1, -1)
    total_nrmse_list = np.array(total_nrmse_list).reshape(1, -1)
    res = np.concatenate((total_f1_micro_list, total_f1_macro_list, total_recall_list, total_precision_list, total_mae_list, total_rmse_list, total_nmae_list, total_nrmse_list))
    np.savetxt(config["data_path"]["result_path"], res, delimiter=",")
