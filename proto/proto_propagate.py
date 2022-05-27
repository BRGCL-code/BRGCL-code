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
from utils import _rank2_trace, _rank2_diag, load_data_from_mat, load_graph, load_data, load_graph_np, visualize_cluster
from GNN import GNNLayer
from evaluation import eva, eva_return
from collections import Counter
import matplotlib.pyplot as plt
from numpy import linalg as LA
import process
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.special import softmax


def evaluation(y, adj, data, model, idx_train, idx_test, out_id=3):
    clf = LogisticRegression(random_state=0, max_iter=2000)
    with torch.no_grad():
        out = model.gae(data, adj)
        embeds = out[out_id]
        train_embs = embeds[idx_train, :] 
        test_embs = embeds[idx_test, :]
        train_labels = torch.Tensor(y[idx_train])
        test_labels = torch.Tensor(y[idx_test])
    clf.fit(train_embs, train_labels)
    pred_test_labels = clf.predict(test_embs)
    return accuracy_score(test_labels, pred_test_labels)


def propagate(adj, alpha, label, class_num, iter_num):
    if label.shape[1] == 1:
        dense_label = np.zeros([label.shape[0], class_num])
        for i in range(label.shape[0]):
            dense_label[i, label[i, 0]] = 1
    else:
        dense_label = label
    
    H = dense_label
    Z = dense_label
    for i in range(iter_num):
        Z = (1 - alpha) * adj * Z + alpha * H
    Z = softmax(Z, axis=1)
    
    return Z


def get_proto_norm(feature, centroid, labels):
    num_data = feature.shape[0]
    each_cluster_num = np.zeros([args.n_clusters])
    for i in range(args.n_clusters):
        each_cluster_num[i] = np.sum(labels==i)
    proto_norm_term = np.zeros([args.n_clusters])
    for i in range(args.n_clusters):
        norm_sum = 0
        for j in range(num_data):
            if labels[j] == i:
                norm_sum = norm_sum + LA.norm(feature[j] - centroid[i], 2)
        proto_norm_term[i] = norm_sum / (each_cluster_num[i] * np.log2(each_cluster_num[i] + 10))

    proto_norm_momt = torch.Tensor(proto_norm_term)
    return proto_norm_momt


def get_proto_loss(feature, centroid, label_momt, proto_norm):
    
    feature_norm = torch.norm(feature, dim=-1)
    feature = torch.div(feature, feature_norm.unsqueeze(1))
    
    centroid_norm = torch.norm(centroid, dim=-1)
    centroid = torch.div(centroid, centroid_norm.unsqueeze(1))
    
    sim_zc = torch.matmul(feature, centroid.t())
    
    sim_zc_normalized = torch.div(sim_zc, proto_norm)
    sim_zc_normalized = torch.exp(sim_zc_normalized)
    sim_2centroid = torch.gather(sim_zc_normalized, -1, label_momt) 
    sim_sum = torch.sum(sim_zc_normalized, -1, keepdim=True)
    sim_2centroid = torch.div(sim_2centroid, sim_sum)
    loss = torch.mean(sim_2centroid.log())
    loss = -1 * loss
    return loss


def train(dataset):
    model = SDCN(500, 500, 500, 500,
            n_input=args.n_input,
            n_z=args.n_z,
            n_clusters=args.n_clusters,
            v=1.0).to(device)
    print(model)

    optimizer = Adam(model.parameters(), lr=args.lr)

    adj_dense, adj = load_graph(args.name, args.k)
    adj_dense_np, adj_np = load_graph_np(args.name, args.k)

    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y

    with torch.no_grad():
        _, _, _, z_momt = model.gae(data, adj)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z_momt.data.cpu().numpy())
    y_pred_last = y_pred
    eva(y, y_pred, 'pae')

    # initialize label_momt
    init_labels = kmeans.labels_
    label_momt = torch.Tensor(init_labels).unsqueeze(1)
    label_momt = label_momt.to(torch.int64)

    # initialize centroid_momt
    ori_center = kmeans.cluster_centers_
    centroid_momt = torch.Tensor(ori_center)

    label_kmeans_ori = kmeans.labels_[:, np.newaxis]

    label_propagated = propagate(adj_np, 0.1, label_kmeans_ori, args.n_clusters, 10)

    centers_propagated = np.dot(label_propagated.T, z_momt) / np.sum(label_propagated.T, axis = 1)[:, np.newaxis]

    label_propagated_hard = np.argmax(label_propagated, axis=1)
    label_propagated_hard = label_propagated_hard[:, np.newaxis]

    label_momt = torch.Tensor(label_propagated_hard)
    label_momt = label_momt.to(torch.int64)

    proto_norm_momt = get_proto_norm(z_momt, ori_center, label_kmeans_ori)

    _, _, _, idx_train, _, idx_test = process.load_data('citeseer')

    best_acc_clf = 0
    best_acc_clt = 0

    for epoch in range(40):    
        x_bar, encoder_out, encoder_out_momt = model(data, adj)

        proto_loss = get_proto_loss(encoder_out, centroid_momt, label_momt, proto_norm_momt)

        loss =  proto_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        with torch.no_grad():
            x_bar, encoder_out, encoder_out_momt = model(data, adj)

            classification_acc = evaluation(y, adj, data, model, idx_train, idx_test, 2)

            print('gnn classification accuracy:' + str(classification_acc))

            if classification_acc > best_acc_clf:
                best_acc_clf = classification_acc
            
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
            y_pred = kmeans.fit_predict(encoder_out_momt.data.cpu().numpy())
            clustering_acc = eva_return(y, kmeans.labels_, str(epoch) + 'encoder')
            if clustering_acc > best_acc_clt:
                best_acc_clt = clustering_acc
        
            label_kmeans = kmeans.labels_[:, np.newaxis]

            label_propagated = propagate(adj_np, 0.1, label_kmeans, args.n_clusters, 10)
            
            centers_propagated = np.dot(label_propagated.T, encoder_out_momt) / np.sum(label_propagated.T, axis = 1)[:, np.newaxis]

            label_propagated_hard = np.argmax(label_propagated, axis=1)
            label_propagated_hard = label_propagated_hard[:, np.newaxis]
            label_momt = torch.Tensor(label_propagated_hard)
            label_momt = label_momt.to(torch.int64)

            centroid_momt = torch.Tensor(centers_propagated)

            proto_norm_momt = get_proto_norm(encoder_out_momt, ori_center, label_kmeans_ori)
            # proto_norm_momt = get_proto_norm(encoder_out_momt, centers_propagated, label_propagated_hard)
            # proto_norm_momt = get_proto_norm(encoder_out_momt, kmeans.cluster_centers_, label_kmeans)

    print('Best gnn classification accuracy: ' + str(best_acc_clf))
    print('Best encoder clustering accuracy: ' + str(best_acc_clt))





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

        self.gae = GAE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_input=n_input,
            n_z=n_z)
        
        self.gae_momt = GAE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_input=n_input,
            n_z=n_z)
        
        self.gae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))
        
        for param_ori, param_momt in zip(self.gae.parameters(), self.gae_momt.parameters()):
            param_momt.data.copy_(param_ori.data)  # initialize
            param_momt.requires_grad = False  # not update by gradient
        
        # degree
        self.v = v
        self.m = 0.9
        
    @torch.no_grad()
    def _momentum_update_momt_encoder(self):
        """
        Momentum update 
        """
        for param_ori, param_momt in zip(self.gae.parameters(), self.gae_momt.parameters()):
            param_momt.data = param_momt.data * self.m + param_ori.data * (1. - self.m)
    
    def forward(self, x, adj):

        x_bar, tra1, tra2, z = self.gae(x, adj)
        
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_momt_encoder()  # update the momentum encoder
            _, _, _, z_momt = self.gae_momt(x, adj)

        return x_bar, z, z_momt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='cite')
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_clusters', default=6, type=int)
    parser.add_argument('--n_z', default=200, type=int)
    parser.add_argument('--pretrain_path', type=str, default='pkl')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    args.cuda = False

    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    args.pretrain_path = 'data/{}.pkl'.format(args.name + '_new_500_500_200_6786_lr5e-4')
    dataset = load_data(args.name)

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
    train(dataset)
