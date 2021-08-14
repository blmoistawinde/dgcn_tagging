#!/bin/bash
# Users can modify the following paths
DATASET_DIR="./audioset_tagging_cnn/raw_dataset"
WORKSPACE="./audioset_tagging_cnn"

GRAPH_DIR='./audioset_all_direct.pkl'
GRAPH_DIR2='./audioset_aser_all_pre_conj_0.pkl'

python3 -u utils/plot_statistics.py plot --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --select=1_full_train_aug_dgcn2 --graph_dir=$GRAPH_DIR --graph_dir2=$GRAPH_DIR2

python3 -u utils/plot_statistics.py plot --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --select=1_full_train_aug_dgcn_inter --graph_dir=$GRAPH_DIR --graph_dir2=$GRAPH_DIR2