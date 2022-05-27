#%%
import time
import argparse
import numpy as np
import torch
from models.GCN import GCN
from models.NRGNN import NRGNN
from dataset import Dataset
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
                    default=False, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=13, help='Random seed.')

parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--edge_hidden', type=int, default=64,
                    help='Number of hidden units of MLP graph constructor')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default="cora", 
                    choices=['cora', 'citeseer','pubmed','dblp'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.2, 
                    help="noise ptb_rate")
parser.add_argument('--epochs', type=int,  default=500, 
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, 
                    help='Initial learning rate.')
parser.add_argument('--alpha', type=float, default=0.03, 
                    help='weight of loss of edge predictor')
parser.add_argument('--beta', type=float, default=1, 
                    help='weight of the loss on pseudo labels')
parser.add_argument('--t_small',type=float, default=0.1, 
                    help='threshold of eliminating the edges')
parser.add_argument('--p_u',type=float, default=0.8, 
                    help='threshold of adding pseudo labels')
parser.add_argument("--n_p", type=int, default=50, 
                    help='number of positive pairs per node')
parser.add_argument("--n_n", type=int, default=50, 
                    help='number of negitive pairs per node')
parser.add_argument("--label_rate", type=float, default=0.05, 
                    help='rate of labeled data')
parser.add_argument('--noise', type=str, default='pair', choices=['uniform', 'pair'], 
                    help='type of noises')

args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
print(device)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print(args)
np.random.seed(15) # Here the random seed is to split the train/val/test data
seed = 12345
np.random.seed(seed)
torch.random.manual_seed(seed)

#%%
if args.dataset=='dblp':
    from torch_geometric.datasets import CitationFull
    import torch_geometric.utils as utils
    dataset = CitationFull('./data','dblp')
    adj = utils.to_scipy_sparse_matrix(dataset.data.edge_index)
    features = dataset.data.x.numpy()
    labels = dataset.data.y.numpy()
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    idx_test = idx[:int(0.8 * len(labels))]
    idx_val = idx[int(0.8 * len(labels)):int(0.9 * len(labels))]
    idx_train = idx[int(0.9 * len(labels)):int((0.9+args.label_rate) * len(labels))]
else:
    data = Dataset(root='./data', name=args.dataset)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_train = idx_train[:int(args.label_rate * adj.shape[0])]

#%% add noise to the labels
from utils import noisify_with_P
ptb = args.ptb_rate
nclass = labels.max() + 1
train_labels = labels[idx_train]
noise_y, P = noisify_with_P(train_labels,nclass, ptb, 10, args.noise) 
noise_labels = labels.copy()
noise_labels[idx_train] = noise_y

# %%
import random
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

dataset = Planetoid(root='/tmp/Cora', name='Cora')


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

loss_CE = torch.nn.CrossEntropyLoss()        
device = torch.device('cpu')
model = GCN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()

for epoch in range(400):
    optimizer.zero_grad()
    out = model(data)
    loss = loss_CE(out[idx_train], data.y[idx_train])
    loss.backward()
    optimizer.step()

model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[idx_test] == data.y[idx_test]).sum()
acc = int(correct) / int(idx_test.sum())
print(f'Accuracy: {acc:.4f}')