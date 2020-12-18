import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import pickle as pkl
from pytorch_lightning.loggers import TensorBoardLogger
import math


class perceptron(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input = nn.Linear(input_size, 128)
        self.linear1 = nn.Linear(128, 128)
        self.output = nn.Linear(128, 1)

    def forward(self, x):
        x = nn.functional.relu(self.input(x.double()))
        x = nn.functional.relu(self.linear1(x))
        x = nn.functional.relu(self.linear1(x))
        x = nn.functional.relu(self.linear1(x))
        out = self.output(x)
        return out


class LevelsetNetwork(pl.LightningModule):
    def __init__(self, input_size):
        super().__init__()
        self.net = perceptron(input_size).double()

    def configure_optimizers(self):
        net_opt = torch.optim.Adam(self.net.parameters(), lr=1.5e-4)
        return net_opt

    def forward(self, x):
        out = self.net(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.net(x)
        loss = F.mse_loss(y_hat, y)
        # log it
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.net(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.net(x)
        loss = F.mse_loss(y_hat, y)
        self.log('test_loss', loss)


class CurvatureData(pl.LightningDataModule):

    def __init__(self, batch_size, data_dir):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir


    def normalize(self, data, stats):
        data_norm = data.clone()
        for i in range(9):
            data_norm[:,i] = (data_norm[:,i] - stats[i, 0]) / stats[i, 1]
        return data_norm

    def setup(self, stage=None):

        stats = pkl.load(open(self.data_dir + '/trainStats.pkl', 'rb'))
        # ======== training dataset ========
        data = pkl.load(open(self.data_dir + '/256_train_data.pkl', 'rb'))
        label = pkl.load(open(self.data_dir + '/256_train_label.pkl', 'rb'))
        data = self.normalize(data, stats)

        self.train_datasets = torch.utils.data.TensorDataset(data, label)
        # ======== validation dataset ========
        data = pkl.load(open(self.data_dir + '/256_val_data.pkl', 'rb'))
        label = pkl.load(open(self.data_dir + '/256_val_label.pkl', 'rb'))
        data = self.normalize(data, stats)

        self.val_datasets = torch.utils.data.TensorDataset(data, label)
        # ======== test dataset ========
        data = pkl.load(open(self.data_dir + '/256_test_data.pkl', 'rb'))
        label = pkl.load(open(self.data_dir + '/256_test_label.pkl', 'rb'))
        data = self.normalize(data, stats)

        self.test_datasets = torch.utils.data.TensorDataset(data, label)



    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_datasets, batch_size=self.batch_size, num_workers=1, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_datasets, batch_size=1, num_workers=1)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_datasets, batch_size=1, num_workers=1)


if __name__ == '__main__':
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath='/home/aho38/m233f20/trial3/checkpoint/weights',
        # save_best_only=True,
        monitor='val_loss',
    )
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=1e-5,
        patience=8,
        verbose=False,
        mode='auto'
    )

    batch_size = 32
    input_size = 9
    data_dir = './data/train'

    # model
    model = LevelsetNetwork(input_size)

    # dataX
    data = CurvatureData(batch_size, data_dir)

    # loggers
    logger = TensorBoardLogger('tb_logs', name='my_model')

    trainer = pl.Trainer(logger=logger, max_epochs=100,
                         # early_stop_callback],
                         callbacks=[checkpoint_callback],
                         gpus=-1
                         )
    trainer.fit(model, data)
