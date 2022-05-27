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
import matplotlib.pyplot as plt
from numpy import linalg as LA

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
        
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        
        self.mlp = torch.nn.Linear(n_z, n_clusters)
        self.softmax = nn.Softmax(dim=-1)
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
        q = self.mlp(z)
        q = self.softmax(q)
        
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_momt_encoder()  # update the momentum encoder
            _, _, _, z_momt = self.gae_momt(x, adj)
        
        return x_bar, q, z, z_momt

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

def get_proto_loss(feature, centroid, label_momt, proto_norm_momt):
    
    feature_norm = torch.norm(feature, dim=-1)
    feature = torch.div(feature, feature_norm.unsqueeze(1))
    
    centroid_norm = torch.norm(centroid, dim=-1)
    centroid = torch.div(centroid, centroid_norm.unsqueeze(1))
    
    sim_zc = torch.matmul(feature, centroid.t())
    
    sim_zc_normalized = torch.div(sim_zc, proto_norm_momt)
    sim_zc_normalized = torch.exp(sim_zc_normalized)
    sim_2centroid = torch.gather(sim_zc_normalized, -1, label_momt) 
    sim_sum = torch.sum(sim_zc_normalized, -1, keepdim=True)
    sim_2centroid = torch.div(sim_2centroid, sim_sum)
    loss = torch.mean(sim_2centroid.log())
    loss = -1 * loss
    return loss

def get_mincut_loss(q, adj_dense):
    out_adj = torch.matmul(torch.matmul(q.transpose(0, 1), adj_dense), q)
    mincut_num = _rank2_trace(out_adj)
    d_flat = torch.einsum('ij->i', adj_dense)
    d = _rank2_diag(d_flat)
    mincut_den = _rank2_trace(torch.matmul(torch.matmul(q.transpose(0, 1), d), q))
    mincut_loss = -(mincut_num / mincut_den)
    mincut_loss = torch.mean(mincut_loss)
    return mincut_loss

def train_sdcn(dataset):
    model = SDCN(500, 2000, 2000, 500,
            n_input=args.n_input,
            n_z=args.n_z,
            n_clusters=args.n_clusters,
            v=1.0).to(device)
    print(model)

    optimizer = Adam(model.parameters(), lr=args.lr)

    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    # KNN Graph
    adj_dense, adj = load_graph(args.name, args.k)

    # adj = adj.cuda()
    # adj_dense = adj_dense.cuda()

    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y
    with torch.no_grad():
        _, _, _, z_momt = model.gae_momt(data, adj)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z_momt.data.cpu().numpy())
    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    
    print(y_pred.shape)
    eva(y, y_pred, 'pae')
    
    
    
    # initialize label_momt
    init_labels = kmeans.labels_
    label_momt = torch.Tensor(init_labels).unsqueeze(1)
    label_momt = label_momt.to(torch.int64)

    # initialize centroid_momt
    ori_center = kmeans.cluster_centers_
    centroid_momt = torch.Tensor(ori_center)
    proto_norm_momt = get_proto_norm(z_momt, centroid_momt, init_labels)
    
    kl_loss_history = []
    ce_loss_history = []
    re_loss_history = []
    mincut_loss_history = []
    proto_loss_history = []
    loss_history = []
    centroid_history = []



    for epoch in range(50):
        
        x_bar, q, encoder_out, encoder_out_momt = model(data, adj)

        mincut_loss = get_mincut_loss(q, adj_dense)
        re_loss = F.mse_loss(x_bar, data)
        proto_loss = get_proto_loss(encoder_out, centroid_momt, label_momt, proto_norm_momt)
        
    #     loss = 0.1 * kl_loss + re_loss + 0.1 * mincut_loss + 0.01 * proto_loss
    #     loss = re_loss + 0.1 * proto_loss + 0.1 * mincut_loss
        loss =  0.1 * proto_loss + 0.1 * mincut_loss
        
        
    #     kl_loss_history.append(kl_loss.data.cpu().detach().numpy())
    #     ce_loss_history.append(ce_loss.data.cpu().detach().numpy())
        re_loss_history.append(re_loss.data.cpu().detach().numpy())
        proto_loss_history.append(proto_loss.data.cpu().detach().numpy())
        mincut_loss_history.append(mincut_loss.data.cpu().detach().numpy())
        loss_history.append(loss.data.cpu().detach().numpy())
    #     centroid_history.append(centroid.data.cpu().detach().numpy())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #     scheduler.step()


        #-----
    #     kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    #     y_pred = kmeans.fit_predict(encoder_out.data.cpu().numpy())

        
    #     init_labels = kmeans.labels_
    #     label_momt = torch.Tensor(init_labels).unsqueeze(1)
    #     label_momt = label_momt.to(torch.int64)
        

    #     ori_center = kmeans.cluster_centers_
    #     centroid_momt = torch.Tensor(ori_center)

    #     proto_norm_momt = get_proto_norm(encoder_out.data.cpu().numpy(), centroid_momt)
        
        
        with torch.no_grad():
            x_bar, q, encoder_out, encoder_out_momt = model(data, adj)
            # p = target_distribution(q.data)
            res1 = q.data.cpu().numpy().argmax(1)       #Q
            # res3 = p.data.cpu().numpy().argmax(1)      #P
            # eva(y, res1, str(epoch) + 'Q')
            # eva(y, res3, str(epoch) + 'P')
            
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
            
            y_pred = kmeans.fit_predict(encoder_out.data.cpu().numpy())
            
            label_momt = torch.Tensor(kmeans.labels_).unsqueeze(1)
            
            label_momt = label_momt.to(torch.int64)
            
            centroid_momt = torch.Tensor(kmeans.cluster_centers_)
            centroid_history.append(centroid_momt.data.cpu().detach().numpy())
            model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
            
            proto_norm_momt = get_proto_norm(encoder_out, centroid_momt, label_momt)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='dblp')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_clusters', default=3, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--pretrain_path', type=str, default='pkl')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    
    args.cuda = False
    
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    args.pretrain_path = 'data/{}.pkl'.format(args.name)
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


    args.pretrain_path = 'data/{}.pkl'.format(args.name + '_new_singlestep_300')

    print(args)
    train_sdcn(dataset)
