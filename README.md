

### BRGCL Code

Sample Code for Bayesian Robust Graph Contrastive Learning

### Abstract 

Graph Neural Networks (GNNs) have been widely used to learn node representations and with outstanding performance on various tasks such as node classification. However, noise, which inevitably exists in real-world graph data, would considerably degrade the performance of GNNs revealed by recent studies. In this work, we propose a novel and robust method, Bayesian Robust Graph Contrastive Learning (BRGCL), which trains a GNN encoder to learn robust node representations. The BRGCL encoder is a completely unsupervised encoder. Two steps are iteratively executed at each epoch of training the BRGCL encoder: (1) estimating confident nodes and computing robust cluster prototypes of node representations through a novel Bayesian nonparametric method; (2) prototypical contrastive learning between the node representations and the robust cluster prototypes. Experiments on public and large-scale benchmarks demonstrate the superior performance of BRGCL and the robustness of the learned node representations.

### Requirements

```
torch==1.7.0
torch-geometric==1.6.1
```

### Dataset

Sample datasets are listed in `./data`

### Train the model

A sample training command on Citeseer

```
python execute.py \
    --name cite \
    --lr 1e-4 \
    --n_input 3703 \
    --noise_level 0.2\
    --gamma0.05 \
    --max_epoch 200 \
    --thres_0 0.2 \
    --noise uniform
```
### Test the model

A sample test command on Citeseer

```
python test.py \
    --name cite \
    --n_input 3703 \
    --noise_level 0.2\
    --thres_0 0.2 \
    --noise uniform \
    --n_clusters 6
```
