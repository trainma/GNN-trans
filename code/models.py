import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import networkx as nx
import numpy as np
import scipy.sparse as sp

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import os
from fbprophet import Prophet
from statsmodels.tsa.arima_model import ARIMA

from PyInstaller.utils.hooks import collect_submodules, collect_data_files
# ----------------------------------------------------------------------------------
from Tmodels import transformer
from numpy.lib.twodim_base import mask_indices
from sklearn.utils import shuffle
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import time
import math
import torch.nn.functional as F
from torch.autograd import Variable
from Tmodels.positional_encoding import *
import argparse
from utils import generate_new_features, generate_new_batches, AverageMeter, generate_batches_lstm, read_meta_datasets

hiddenimports = collect_submodules('fbprophet')
datas = collect_data_files('fbprophet')


def arima(ahead, start_exp, n_samples, labels):
    var = []
    for idx in range(ahead):
        var.append([])

    error = np.zeros(ahead)
    count = 0
    for test_sample in range(start_exp, n_samples - ahead):  #
        print(test_sample)
        count += 1
        err = 0
        for j in range(labels.shape[0]):
            ds = labels.iloc[j, :test_sample - 1].reset_index()

            if (sum(ds.iloc[:, 1]) == 0):
                yhat = [0] * (ahead)
            else:
                try:
                    fit2 = ARIMA(ds.iloc[:, 1].values, (2, 0, 2)).fit()
                except:
                    fit2 = ARIMA(ds.iloc[:, 1].values, (1, 0, 0)).fit()
                # yhat = abs(fit2.predict(start = test_sample , end = (test_sample+ahead-1) ))
                yhat = abs(fit2.predict(start=test_sample, end=(test_sample + ahead - 2)))
            y_me = labels.iloc[j, test_sample:test_sample + ahead]
            e = abs(yhat - y_me.values)
            err += e
            error += e

        for idx in range(ahead):
            var[idx].append(err[idx])
    return error, var


def prophet(ahead, start_exp, n_samples, labels):
    var = []
    for idx in range(ahead):
        var.append([])

    error = np.zeros(ahead)
    count = 0
    for test_sample in range(start_exp, n_samples - ahead):  #
        print(test_sample)
        count += 1
        err = 0
        for j in range(labels.shape[0]):
            ds = labels.iloc[j, :test_sample].reset_index()
            ds.columns = ["ds", "y"]
            # with suppress_stdout_stderr():
            m = Prophet(interval_width=0.95)
            m.fit(ds)
            future = m.predict(m.make_future_dataframe(periods=ahead))
            yhat = future["yhat"].tail(ahead)
            y_me = labels.iloc[j, test_sample:test_sample + ahead]
            e = abs(yhat - y_me.values).values
            err += e
            error += e
        for idx in range(ahead):
            var[idx].append(err[idx])

    return error, var


class MPNN_LSTM(nn.Module):
    def __init__(self, nfeat, nhid, nout, n_nodes, window, dropout):
        super(MPNN_LSTM, self).__init__()
        self.window = window
        self.n_nodes = n_nodes
        # self.batch_size = batch_size
        self.nhid = nhid
        self.nfeat = nfeat
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nhid)

        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)

        self.rnn1 = nn.LSTM(2 * nhid, nhid, 1)
        self.rnn2 = nn.LSTM(nhid, nhid, 1)

        # self.fc1 = nn.Linear(2*nhid+window*nfeat, nhid)
        self.fc1 = nn.Linear(2 * nhid + window * nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nout)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, adj, x):
        lst = list()
        # print("--------------------")
        weight = adj.coalesce().values()
        adj = adj.coalesce().indices()
        # print(x.shape)
        # [bz,wd,n_nodes,feats]
        skip = x.view(-1, self.window, self.n_nodes, self.nfeat)  # self.batch_size
        # print(skip.shape)
        # [645 7 7]
        skip = torch.transpose(skip, 1, 2).reshape(-1, self.window, self.nfeat)  # self.batch_size*self.n_nodes
        # x [4515 64]
        x = self.relu(self.conv1(x, adj, edge_weight=weight))
        x = self.bn1(x)
        x = self.dropout(x)
        lst.append(x)

        x = self.relu(self.conv2(x, adj, edge_weight=weight))
        x = self.bn2(x)
        x = self.dropout(x)
        lst.append(x)
        # x : [4515 64]
        x = torch.cat(lst, dim=1)  # x [4515 128]

        # --------------------------------------
        # print(x.shape)
        x = x.view(-1, self.window, self.n_nodes, x.size(1))  # torch.Size([5, 7, 129, 128])
        # print(x.shape)
        # print(x.shape)
        x = torch.transpose(x, 0, 1)  # torch.Size([7, 5, 129, 128])
        x = x.contiguous().view(self.window, -1, x.size(3))  # self.batch_size*self.n_nodes[7 5*129 128]

        # print(x.shape)
        # print("------")
        x, (hn1, cn1) = self.rnn1(x)  # [7(seq_len-time windows) 5*129(batch_size) 128(d_models)]

        out2, (hn2, cn2) = self.rnn2(x)

        # print(self.rnn2._all_weights)
        x = torch.cat([hn1[0, :, :], hn2[0, :, :]], dim=1)
        # print(skip.shape)
        # print(x.shape)
        # skip = skip.view(skip.size(0),-1)
        skip = skip.reshape(skip.size(0), -1)
        # print(x.shape)
        # print(skip.shape)

        x = torch.cat([x, skip], dim=1)
        # --------------------------------------
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x)).squeeze()
        x = x.view(-1)
        # print("--------------------")

        return x


class MPNN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(MPNN, self).__init__()
        # self.n_nodes = n_nodes

        # self.batch_size = batch_size
        self.nhid = nhid

        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nhid)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)

        self.fc1 = nn.Linear(nfeat + 2 * nhid, nhid)
        self.fc2 = nn.Linear(nhid, nout)
        # self.bn3 = nn.BatchNorm1d(nhid)
        # self.bn4 = nn.BatchNorm1d(nhid)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # nn.init.zeros_(self.conv1.weight)
        # nn.init.zeros_(self.conv2.weight)
        # nn.init.zeros_(self.fc1.weight)
        # nn.init.zeros_(self.fc2.weight)

    def forward(self, adj, x):
        lst = list()
        # print(x.shape)
        # print(adj.shape)
        # x [272 7]
        # adj [272 272]
        weight = adj.coalesce().values()
        adj = adj.coalesce().indices()

        # lst.append(ident)

        # x = x[:,mob_feats]
        # x = xt.index_select(1, mob_feats)
        lst.append(x)
        # (272 64)
        x = self.relu(self.conv1(x, adj, edge_weight=weight))
        # print(x)
        # print(x.shape)
        x = self.bn1(x)
        x = self.dropout(x)
        lst.append(x)
        # x (272 64) adj(2 9248) weight(9248)
        x = self.relu(self.conv2(x, adj, edge_weight=weight))
        # print(x.shape)
        x = self.bn2(x)
        x = self.dropout(x)
        lst.append(x)

        x = torch.cat(lst, dim=1)

        x = self.relu(self.fc1(x))
        # x = self.bn3(x)
        x = self.dropout(x)

        x = self.relu(self.fc2(x)).squeeze()  #
        # x = self.bn4(x)

        x = x.view(-1)

        return x


class LSTM(nn.Module):
    def __init__(self, nfeat, nhid, n_nodes, window, dropout, batch_size, recur):
        super().__init__()
        self.nhid = nhid
        self.n_nodes = n_nodes
        self.nout = n_nodes
        self.window = window
        self.nb_layers = 2

        self.nfeat = nfeat
        self.recur = recur
        self.batch_size = batch_size
        self.lstm = nn.LSTM(nfeat, self.nhid, num_layers=self.nb_layers)

        self.linear = nn.Linear(nhid, self.nout)
        self.cell = (nn.Parameter(nn.init.xavier_uniform(
            torch.Tensor(self.nb_layers, self.batch_size, self.nhid).type(torch.FloatTensor).cuda()),
            requires_grad=True))

        # self.hidden_cell = (torch.zeros(2,self.batch_size,self.nhid).to(device),torch.zeros(2,self.batch_size,self.nhid).to(device))
        # nn.Parameter(nn.init.xavier_uniform(torch.Tensor(self.nb_layers, self.batch_size, self.nhid).type(torch.FloatTensor).cuda()),requires_grad=True))

    def forward(self, adj, features):
        # adj is 0 here
        # print(features.shape)
        features = features.view(self.window, -1, self.n_nodes)  # .view(-1, self.window, self.n_nodes, self.nfeat)
        # print(features.shape)
        # print("----")

        # ------------------
        if (self.recur):
            # print(features.shape)
            # self.hidden_cell =
            try:
                lstm_out, (hc, self.cell) = self.lstm(features, (
                    torch.zeros(self.nb_layers, self.batch_size, self.nhid).cuda(), self.cell))
                # = (hc,cn)
            except:
                # hc = self.hidden_cell[0][:,0:features.shape[1],:].contiguous().view(2,features.shape[1],self.nhid)
                hc = torch.zeros(self.nb_layers, features.shape[1], self.nhid).cuda()
                cn = self.cell[:, 0:features.shape[1], :].contiguous().view(2, features.shape[1], self.nhid)
                lstm_out, (hc, cn) = self.lstm(features, (hc, cn))
        else:
            # ------------------
            lstm_out, (hc, cn) = self.lstm(features)  # , self.hidden_cell)#self.hidden_cell

        predictions = self.linear(lstm_out)  # .view(self.window,-1,self.n_nodes)#.view(self.batch_size,self.nhid))#)
        # print(predictions.shape)
        return predictions[-1].view(-1)


class MPNN_trans(nn.Module):
    def __init__(self, nfeat, nhid, nout, n_nodes, window, dropout, d_model=256, num_layers=3, position='fixed'):
        super(MPNN_trans, self).__init__()
        self.window = window
        self.n_nodes = n_nodes
        # self.batch_size = batch_size
        self.nhid = nhid
        self.nfeat = nfeat
        self.d_model = d_model

        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nhid)

        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.input_project = nn.Linear(nhid * 2, self.d_model)
        self.local = context_embedding(self.d_model, self.d_model, 1)
        self.pos_encoder = PositionalEncoding(self.d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=16, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=16, dropout=dropout)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, 1)

        self.tmp_out = nn.Linear(self.d_model, 128)
        self.tmp_out2 = nn.Linear(128, 1)
        self.src_key_padding_mask = None
        self.device = torch.device("cuda")
        self.transformer = nn.Transformer(d_model=self.d_model, nhead=8, num_encoder_layers=num_layers,
                                          num_decoder_layers=1, dropout=dropout)

    def forward(self, adj, x):
        lst = list()
        # print("--------------------")
        weight = adj.coalesce().values()
        adj = adj.coalesce().indices()
        # print(x.shape)
        # [bz,wd,n_nodes,feats]
        skip = x.view(-1, self.window, self.n_nodes, self.nfeat)  # self.batch_size
        # print(skip.shape)
        # [645 7 7]
        skip = torch.transpose(skip, 1, 2).reshape(-1, self.window, self.nfeat)  # self.batch_size*self.n_nodes
        # x [4515 64]
        x = self.relu(self.conv1(x, adj, edge_weight=weight))
        x = self.bn1(x)
        x = self.dropout(x)
        lst.append(x)

        x = self.relu(self.conv2(x, adj, edge_weight=weight))
        x = self.bn2(x)
        x = self.dropout(x)
        lst.append(x)
        # x : [4515 64]
        x = torch.cat(lst, dim=1)  # x [4515 128]

        x = x.view(-1, self.window, self.n_nodes, x.size(1))
        x = torch.transpose(x, 0, 1)
        x = x.contiguous().view(self.window, -1, x.size(3))
        # x [window bz feature_dim]
        # --TRANSFORMER_OUT src[7 645 128] [windowsize bz feature_dim]
        src = x
        # src = src.permute(1, 0, 2)  # [bz windowsize feature_dim]
        tgt = src[-3:, :, :]
        mask = self._generate_square_subsequent_mask(tgt.shape[0]).to(self.device)

        src = self.input_project(src) * math.sqrt(self.d_model)
        tgt = self.input_project(tgt) * math.sqrt(self.d_model)

        src = self.local(src.permute(1, 2, 0))
        src = src.permute(2, 0, 1)
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        x = self.transformer(src=src, tgt=tgt, tgt_mask=mask)
        transformer_out = self.relu(self.tmp_out(x))
        transformer_out = self.tmp_out2(transformer_out)[0, :, :]

        return transformer_out.squeeze()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Size of batch.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate.')
    parser.add_argument('--window', type=int, default=7,
                        help='Size of window for features.')
    parser.add_argument('--graph-window', type=int, default=7,
                        help='Size of window for graphs in MPNN LSTM.')
    parser.add_argument('--recur', default=False,
                        help='True or False.')
    parser.add_argument('--early-stop', type=int, default=100,
                        help='How many epochs to wait before stopping.')
    parser.add_argument('--start-exp', type=int, default=15,
                        help='The first day to start the predictions.')
    parser.add_argument('--ahead', type=int, default=14,
                        help='The number of days ahead of the train set the predictions should reach.')
    parser.add_argument('--sep', type=int, default=10,
                        help='Seperator for validation and train set.')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
    # read Dataset from data which is in ./data/{countries}
    meta_labs, meta_graphs, meta_features, meta_y = read_meta_datasets(args.window)

    for country in ["IT", "ES", "FR", "EN"]:  # ,",
        if (country == "IT"):
            idx = 0

        elif (country == "ES"):
            idx = 1

        elif (country == "EN"):
            idx = 2
        else:
            idx = 3

        labels = meta_labs[idx]
        gs_adj = meta_graphs[idx]
        features = meta_features[idx]
        y = meta_y[idx]
        n_samples = len(gs_adj)
        nfeat = meta_features[0][0].shape[1]

        n_nodes = gs_adj[0].shape[0]
        print(n_nodes)
        if not os.path.exists('../results'):
            os.makedirs('../results')
    fw = open("../results/results_" + country + ".csv", "a")

    lstm_features = 1 * n_nodes
    idx_train = [6, 8, 10, 12, 14]
    shift = 0
    test_sample = 15
    adj_train, features_train, y_train = generate_new_batches(gs_adj, features, y, idx_train,
                                                              args.graph_window, shift, args.batch_size,
                                                              device, test_sample)
    n_train_batches = math.ceil(len(idx_train) / args.batch_size)
    n_val_batches = 1
    n_test_batches = 1
    model = MPNN_trans(nfeat=nfeat, nhid=args.hidden, nout=1, n_nodes=n_nodes,
                       window=args.graph_window, dropout=args.dropout, num_layers=3).to(device)
    model2 = MPNN_LSTM(nfeat=nfeat, nhid=args.hidden, nout=1, n_nodes=n_nodes,
                       window=args.graph_window, dropout=args.dropout).to(device)
    tmp = model.forward(adj_train[0], features_train[0])
