

### BRGCL Code

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
