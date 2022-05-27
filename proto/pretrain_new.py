import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.nn import Linear
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from evaluation import eva, eva_return
from torch.nn.modules.module import Module
from utils import load_graph
from GNN import GNNLayer

# python pretrain_new.py
#torch.cuda.set_device(3)

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

        return x_bar, z


class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001 * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def pretrain_ae(model, dataset, y):
    # train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    print(model)
    x = torch.Tensor(dataset.x)
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=1.0)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 300], gamma=0.1)
    for epoch in range(50):
        # adjust_learning_rate(optimizer, epoch)
        # for batch_idx, (x, _) in enumerate(train_loader):
        
        x = x.cuda()

        x_bar, _ = model(x, adj)
        loss = F.mse_loss(x_bar, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            x = torch.Tensor(dataset.x).float().cuda()
            x_bar, z = model(x, adj)
            loss = F.mse_loss(x_bar, x)
            print('{} loss: {}'.format(epoch, loss))           
            kmeans = KMeans(n_clusters=4, n_init=20).fit(z.data.cpu().numpy())
            eva(y, kmeans.labels_, epoch)

        torch.save(model.state_dict(), 'data/dblp_new_singlestep_50.pkl')

model = GAE(
        n_enc_1=500,
        n_enc_2=2000,
        n_dec_1=2000,
        n_dec_2=500,
        n_input=334,
        n_z=10,).cuda()

x = np.loadtxt('data/dblp.txt', dtype=float)
y = np.loadtxt('data/dblp_label.txt', dtype=int)

dataset = LoadDataset(x)
_, adj = load_graph("dblp", None)
adj = adj.cuda()
pretrain_ae(model, dataset, y)
