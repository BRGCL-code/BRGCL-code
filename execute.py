from __future__ import print_function, division
import argparse
from re import X
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import Linear
from utils import load_graph, load_data
from GNN import GNNLayer
from numpy import linalg as LA
from numpy.testing import assert_array_almost_equal
import process
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from scipy.special import softmax

def build_uniform_P(size, noise):
    """ The noise matrix flips any class to any other with probability
    noise / (#class - 1).
    """

    assert(noise >= 0.) and (noise <= 1.)

    P = np.float64(noise) / np.float64(size - 1) * np.ones((size, size))
    np.fill_diagonal(P, (np.float64(1)-np.float64(noise))*np.ones(size))
    
    diag_idx = np.arange(size)
    P[diag_idx,diag_idx] = P[diag_idx,diag_idx] + 1.0 - P.sum(0)
    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P

def build_pair_p(size, noise):
    assert(noise >= 0.) and (noise <= 1.)
    P = (1.0 - np.float64(noise)) * np.eye(size)
    for i in range(size):
        P[i,i-1] = np.float64(noise)
    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P

def cross_entropy(y, y_0):
    result=-np.sum(y*np.log(y_0))
    return result/float(y_0.shape[0])

def estimate_conf(l, l_0):
    conf = np.zeros([l.shape[0]])
    for i in range(l.shape[0]):
        conf[i] = cross_entropy(l[i,:],l_0[i,:])


def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """

    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y

def noisify_with_P(y_train, nb_classes, noise, random_state=None,  noise_type='uniform'):
    
    if noise > 0.0:
        if noise_type=='uniform':
            print('Uniform noise')
            P = build_uniform_P(nb_classes, noise)
        elif noise_type == 'pair':
            print('Pair noise')
            P = build_pair_p(nb_classes, noise)
        else:
            print('Noise type have implemented')
        # seed the random numbers with #run
        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    else:
        P = np.eye(nb_classes)

    return y_train, P

class DP:
    def __init__(self,X):                    
        self.K = 1
        self.d = X.shape[1]
        self.z = np.mod(np.random.permutation(X.shape[0]),self.K)+1
        self.mu = np.random.standard_normal((self.K, self.d))
        self.sigma = 1
        self.nk = np.zeros(self.K)
        self.pik = np.ones(self.K)/self.K 

        self.mu = np.array([np.mean(X,0)])
        self.Lambda = 0.05
        self.max_iter = 10
        self.obj = np.zeros(self.max_iter)
        self.em_time = np.zeros(self.max_iter)   
        
    def fit(self,X):
        max_iter = self.max_iter        
        [n,d] = np.shape(X)      
        for iter in range(max_iter):
            dist = np.zeros((n,self.K))
            for kk in range(self.K):
                Xm = X - np.tile(self.mu[kk,:],(n,1))
                dist[:,kk] = np.sum(Xm*Xm,1)            
            dmin = np.min(dist,1)
            self.z = np.argmin(dist,1)
            idx = np.where(dmin > self.Lambda)
            
            if (np.size(idx) > 0):
                self.K = self.K + 1
                self.z[idx[0]] = self.K-1 
                self.mu = np.vstack([self.mu,np.mean(X[idx[0],:],0)])                
                Xm = X - np.tile(self.mu[self.K-1,:],(n,1))
                dist = np.hstack([dist, np.array([np.sum(Xm*Xm,1)]).T])
            self.nk = np.zeros(self.K)
            for kk in range(self.K):
                self.nk[kk] = self.z.tolist().count(kk)
                idx = np.where(self.z == kk)
                self.mu[kk,:] = np.mean(X[idx[0],:],0)
            
            self.pik = self.nk/float(np.sum(self.nk))

        return self.z, self.K
    

def evaluation(adj, data, model, idx_train, idx_test, y, noise_level):
    clf_MLP = MLPClassifier(random_state=1, max_iter=800, hidden_layer_sizes=(250,))
    with torch.no_grad():
        _,_,embeds = model.gcn(data, adj)
        train_embs = embeds[idx_train, :] 
        test_embs = embeds[idx_test, :]
        num_classes = 6
        train_label_only = y[idx_train]-1
        ptb = noise_level
        noise_type = 'uniform'
        noise_y, P = noisify_with_P(train_label_only, num_classes, noise_level, 10, noise_type) 
        train_labels = torch.Tensor(noise_y+1)
        test_labels = torch.Tensor(y[idx_test])

    clf_MLP.fit(train_embs, train_labels)
    pred_test_labels = clf_MLP.predict(test_embs)
    return accuracy_score(test_labels, pred_test_labels), pred_test_labels

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

class GCN_encoder(nn.Module):
    
    def __init__(self, n_enc_1, n_enc_2,
                 n_input, n_z):
        super(GCN_encoder, self).__init__()
        self.enc_1 = GNNLayer(n_input, n_enc_1)
        self.enc_2 = GNNLayer(n_enc_1, n_enc_2)
        self.z_layer = Linear(n_enc_2, n_z)

    def forward(self, x, adj):
        enc_h1 = F.relu(self.enc_1(x, adj))
        enc_h2 = F.relu(self.enc_2(enc_h1, adj))
        z = self.z_layer(enc_h2)

        return enc_h1, enc_h2, z

class BRGCL(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_input, n_z):
        super(BRGCL, self).__init__()

        # autoencoder for intra information
        self.gcn = GCN_encoder(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_input=n_input,
            n_z=n_z)
        
        self.gcn.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))
    
    def forward(self, x, adj):

        tra1, tra2, z = self.gcn(x, adj)
        return tra2, z

def get_proto_norm(feature, centroid, ps_label):
    num_data = feature.shape[0]
    each_cluster_num = np.zeros([args.n_clusters])
    for i in range(args.n_clusters):
        each_cluster_num[i] = np.sum(ps_label==i)
    proto_norm_term = np.zeros([args.n_clusters])
    for i in range(args.n_clusters):
        norm_sum = 0
        for j in range(num_data):
            if ps_label[j] == i:
                norm_sum = norm_sum + LA.norm(feature[j] - centroid[i], 2)
        proto_norm_term[i] = norm_sum / (each_cluster_num[i] * np.log2(each_cluster_num[i] + 10))
    proto_norm = torch.Tensor(proto_norm_term)
    return proto_norm

def get_proto_loss(feature, centroid, ps_label, proto_norm):
    
    feature_norm = torch.norm(feature, dim=-1)
    feature = torch.div(feature, feature_norm.unsqueeze(1))
    
    centroid_norm = torch.norm(centroid, dim=-1)
    centroid = torch.div(centroid, centroid_norm.unsqueeze(1))
    
    sim_zc = torch.matmul(feature, centroid.t())
    
    sim_zc_normalized = torch.div(sim_zc, proto_norm)
    sim_zc_normalized = torch.exp(sim_zc_normalized)
    sim_2centroid = torch.gather(sim_zc_normalized, -1, ps_label) 
    sim_sum = torch.sum(sim_zc_normalized, -1, keepdim=True)
    sim_2centroid = torch.div(sim_2centroid, sim_sum)
    loss = torch.mean(sim_2centroid.log())
    loss = -1 * loss
    return loss

def train(args):
    _, _, _, idx_train, _, idx_test = process.load_data(args.name)
    _, adj = load_graph(args.name, args.k)
    dataset = load_data(args.name)
    model = BRGCL(500, 500, n_input=args.n_input, n_z=args.n_z).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y
    best_acc_clf = 0
    y_predicts_clf = []
    thres_0 = args.thres_0

    with torch.no_grad():
        _, _, encoder_out = model(data, adj)
        DP_model = DP(encoder_out)
        estimated_K, ps_labels = DP_model.fit(X)
        kmeans = KMeans(n_clusters=estimated_K, n_init=20)
        ori_center = kmeans.cluster_centers_
        centers = torch.Tensor(ori_center)    
        proto_norm = get_proto_norm(encoder_out, centers)
        confident_centers = centers
    for epoch in range(args.max_epoch):    
        _, _, encoder_out = model(data, adj)
        loss = get_proto_loss(encoder_out, confident_centers, ps_labels, proto_norm)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            _, _, encoder_out = model(data, adj)
            
            classification_acc, pred_test_clf = evaluation(adj, data, model, idx_train, idx_test, y)
            y_predicts_clf.append(pred_test_clf)
            print('gnn classification accuracy:' + str(classification_acc))
            if classification_acc > best_acc_clf:
                best_acc_clf = classification_acc

            kmeans = KMeans(n_clusters=estimated_K, n_init=20)
            label_kmeans = kmeans.labels_[:, np.newaxis]
            dense_ps_labels = np.zeros([label_kmeans.shape[0], estimated_K])
            for i in range(label_kmeans.shape[0]):
                dense_ps_labels[i, label_kmeans[i, 0]] = 1
            label_propagated = propagate(adj, 0.1, label_kmeans, estimated_K, 10)
            confidence = estimate_conf(dense_ps_labels, label_propagated)
            ps_labels = label_kmeans
            confident_labels = ps_labels
            thres = 1-np.min(thres_0, thres_0*(epoch/args.epoch_m))
            confident_labels[confidence>thres, :] = 0
            confident_centers = np.dot(confident_labels.T, encoder_out) / np.sum(confident_labels.T, axis = 1)[:, np.newaxis]

    print('Best gnn classification accuracy: ' + str(best_acc_clf))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='citeseer')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_epoch', type=int, default=400)
    parser.add_argument('--n_clusters', default=3, type=int)
    parser.add_argument('--n_input', default=3703, type=int)
    parser.add_argument('--thres_0', default=0.3, type=float)
    parser.add_argument('--gamma', default=0.05, type=float)
    parser.add_argument('--pretrain_path', type=str, default='pkl')
    parser.add_argument('--noise_level', type=float, default=0.2)
    parser.add_argument('--noise', type=str, default='uniform', choices=['uniform', 'pair'])
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")
    args.pretrain_path = 'models/{}.pkl'.format(args.name)
    args.name == 'cite'
    args.lr = 1e-4
    args.thres_0 = 0.3
    args.max_epoch = 60
    args.epoch_m = int(0.9 * 60)
    args.n_clusters = 6
    args.n_input = 3703
    train(args)
