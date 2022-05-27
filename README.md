

## Requirements

```
torch==1.4.0
torch-geometric==1.6.1
```

## Run the code
```
python train_robust.py \
    --dataset cora \
    --seed 11 \
    --t_small 0.1 \
    --alpha 0.03\
    --lr 0.001 \
    --epochs 200 \
    --n_p -1 \
    --p_u 0.8 \
    --label_rate 0.05 \
    --ptb_rate 0.2 \
    --noise uniform
```
## Dataset
Datasets are listed in `./data`

