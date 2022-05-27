from __future__ import print_function, division
import argparse
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from utils import _rank2_trace, _rank2_diag, load_data_from_mat, load_graph, load_data
from GNN import GNNLayer
from evaluation import eva
from collections import Counter


# torch.cuda.set_device(1)


class GAE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_dec_1, n_dec_2,
                 n_input, n_z):
        super(GAE, self).__init__()
        self.enc_1 = GNNLayer(n_input, n_enc_1)
        self.enc_2 = GNNLayer(n_enc_1, n_enc_2)
        self.z_layer = Linear(n_enc_2, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.x_bar_layer = Linear(n_dec_2, n_input)

    def forward(self, x, adj):
        enc_h1 = F.relu(self.enc_1(x, adj))
        enc_h2 = F.relu(self.enc_2(enc_h1, adj))
        z = self.z_layer(enc_h2)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        x_bar = self.x_bar_layer(dec_h2)

        return x_bar, enc_h1, enc_h2, z


class SDCN(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_dec_1, n_dec_2,
                 n_input, n_z, n_clusters, v=1):
        super(SDCN, self).__init__()

        # autoencoder for intra information
        self.gae = GAE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_input=n_input,
            n_z=n_z)
        # self.gae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, tra2, z = self.gae(x, adj)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, z


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_sdcn(dataset):
    model = SDCN(64, 32, 32, 64,
                 n_input=args.n_input,
                 n_z=args.n_z,
                 n_clusters=args.n_clusters,
                 v=1.0).to(device)
    print(model)

    optimizer = Adam(model.parameters(), lr=args.lr)

    # KNN Graph
    data, adj, y = load_data_from_mat(dataset)
    # adj_dense, adj = load_graph(args.name, args.k)

    adj = adj.cuda()
    data = data.cuda()

    # cluster parameter initiate
    # data = torch.Tensor(dataset.x).to(device)
    # y = dataset.y
    with torch.no_grad():
        _, _, _, z = model.gae(data, adj)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, y_pred, 'pae')

    for epoch in range(200):
        if epoch % 1 == 0:
            # update_interval
            _, tmp_q, _ = model(data, adj)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

            res1 = tmp_q.cpu().numpy().argmax(1)  # Q
            # res2 = pred.data.cpu().numpy().argmax(1)   #Z
            res3 = p.data.cpu().numpy().argmax(1)  # P
            eva(y, res1, str(epoch) + 'Q')
            # eva(y, res2, str(epoch) + 'Z')
            eva(y, res3, str(epoch) + 'P')

        x_bar, q, _ = model(data, adj)

        # MinCUT regularization.
        # out_adj = torch.matmul(torch.sparse.mm(q.transpose(0, 1), adj), q)
        # mincut_num = _rank2_trace(out_adj)
        # d_flat = torch.einsum('ij->i', adj_dense)
        # d = _rank2_diag(d_flat)
        # mincut_den = _rank2_trace(
        #     torch.matmul(torch.matmul(q.transpose(0, 1), d), q))
        # mincut_loss = -(mincut_num / mincut_den)
        # mincut_loss = torch.mean(mincut_loss)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        # ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)

        # loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss
        # loss = 0.1 * kl_loss + 0.1 * mincut_loss + re_loss
        loss = 0.1 * kl_loss + re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='ogbn-arxiv')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_clusters', default=3, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--pretrain_path', type=str, default='pkl')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    args.pretrain_path = 'data/{}.pkl'.format(args.name)
    # dataset = load_data(args.name)

    if args.name == 'ogbn-arxiv':
        args.n_clusters = 40
        args.n_input = 128

    if args.name == 'usps':
        args.n_clusters = 10
        args.n_input = 256

    if args.name == 'hhar':
        args.k = 5
        args.n_clusters = 6
        args.n_input = 561

    if args.name == 'reut':
        args.lr = 1e-4
        args.n_clusters = 4
        args.n_input = 2000

    if args.name == 'acm':
        args.k = None
        args.n_clusters = 3
        args.n_input = 1870

    if args.name == 'dblp':
        args.k = None
        args.n_clusters = 4
        args.n_input = 334

    if args.name == 'cite':
        args.lr = 1e-4
        args.k = None
        args.n_clusters = 6
        args.n_input = 3703

    print(args)
    train_sdcn(args.name)
