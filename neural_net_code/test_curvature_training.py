import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import pickle as pkl
from pytorch_lightning.loggers import TensorBoardLogger
import math
import matplotlib.pyplot as plt

from curvature_training import LevelsetNetwork, CurvatureData, perceptron

def normalize(data, stats):
    data_norm = data.clone()
    for i in range(9):
        data_norm[:,i] = (data_norm[:,i] - stats[i, 0]) / stats[i, 1]
    # for i in range(9):
    #     data_norm[:,i] = (data_norm[:,i] - data_norm[:,i].mean()) / data_norm[:,i].std()
    return data_norm

def get_stats():
    stats = pkl.load(open('./data/train/trainStats.pkl', 'rb'))
    return stats


if __name__ == '__main__':
    input_size = 9
    # data_dir = './data/266_data.pkl'
    # label_dir = './data/266_label.pkl'

    # model
    model = LevelsetNetwork(input_size).load_from_checkpoint('./checkpoint/weights-v0.ckpt', input_size=input_size)

    # dataX
    #data = CurvatureData(1, './data/test_circle/exp2531_data.pkl', './data/test_circle/exp2531_label.pkl')
    x = pkl.load(open('./data/test_petal/20_reinit_test_petal_data.pkl', 'rb'))
    y = pkl.load(open('./data/test_petal/20_reinit_test_petal_label.pkl', 'rb'))

    x = normalize(x, get_stats())

    data = torch.utils.data.TensorDataset(x, y)
    dataload = torch.utils.data.DataLoader(data, batch_size=1)

    ls_y_hat = []
    ls_y = []
    loss_ls = []

    stats = get_stats()
    print(stats)

    # loggers

    model.eval()
    torch.set_grad_enabled(False)

    with torch.no_grad():
        for i, data in enumerate(dataload):
            x, y = data
            # print(x)



            y_hat = model(x)

            #print('loss: ', F.mse_loss(y, y_hat))
            # if  F.mse_loss(y, y_hat) > 0.0001:
            #     continue
            print(F.mse_loss(y, y_hat))
            loss_ls.append(F.mse_loss(y, y_hat))

            ls_y_hat.append(torch.tensor([y_hat]))
            ls_y.append(torch.tensor([y]))




    tensor_y_hat = torch.stack(ls_y_hat, 0)
    tensor_y = torch.stack(ls_y,0)
    pkl.dump(tensor_y_hat, open('./result/petal/20_reinit_tensor_y_hat.pkl', 'wb'))
    pkl.dump(tensor_y, open('./result/petal/20_reinit_tensor_y.pkl', 'wb'))

    plt.plot([i for i in range(len(loss_ls))], loss_ls)
    plt.show()
