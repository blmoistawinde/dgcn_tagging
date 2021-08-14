#!/bin/bash
# Users can modify the following paths
DATASET_DIR="./data"
WORKSPACE="./workspaces"

# Plot statistics
python3 utils/plot_statistics.py plot --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --select=1_full_train_5_aug

GRAPH_DIR='./audioset_all_direct.pkl'

python3 utils/plot_statistics.py plot --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --select=1_full_train_5_gcn_aug --graph_dir=$GRAPH_DIR

GRAPH_DIR='./aser_all_pre_conj_0.pkl'

python3 utils/plot_statistics.py plot --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --select=1_full_train_5_gcn_aug --graph_dir=$GRAPH_DIR

GRAPH_DIR='./ATGCN_graph.pkl'

python3 utils/plot_statistics.py plot --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --select=1_full_train_5_gcn_aug --graph_dir=$GRAPH_DIR

GRAPH_DIR='./audioset_aser_all_pre_conj_0.pkl'

python3 utils/plot_statistics.py plot --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --select=1_full_train_5_gcn_aug --graph_dir=$GRAPH_DIR

GRAPH_DIR='./audioset_all_direct.pkl'
GRAPH_DIR2='./aser_all_pre_conj_0.pkl'

python3 -u utils/plot_statistics.py plot --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --select=1_full_train_5_dgcn_inter_aug --graph_dir=$GRAPH_DIR --graph_dir2=$GRAPH_DIR2