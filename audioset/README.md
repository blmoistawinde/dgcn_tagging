# AudioSet experimental codes

Mainly modified from https://github.com/qiuqiangkong/audioset_tagging_cnn

1. refer to the original repo to download and pack the data into hdf5s

2. run `runme_*.sh` to start the experiments
    - "1p", "5p", "10p" means using 1%, 5%, 10% of the dataset, needs to run `sample_indices.py` to sample these subsets first
    - "gcn1", "gcn2", "gcn3" uses different graph as the backbone of GCN based model
    - "atgcn1" reimplements the AT-GCN method
    - "dgcn" is our proposed Double-GCN

3. use `plot_*.sh` to draw the plots and get the precise experimental results