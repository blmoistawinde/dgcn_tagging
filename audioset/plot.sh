#!/bin/bash
# Users can modify the following paths
DATASET_DIR="./audioset_tagging_cnn/raw_dataset"
WORKSPACE="./audioset_tagging_cnn"

# Plot statistics
python3 utils/plot_statistics.py plot --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --select='1_full_train'

GRAPH_DIR='./audioset_aser_all_pre_conj_0.pkl'

python3 utils/plot_statistics.py plot --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --select='1_full_train_gcn' --graph_dir=$GRAPH_DIR

GRAPH_DIR='./aser_all_pre_conj_0.pkl'

python3 utils/plot_statistics.py plot --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --select='1_full_train_gcn' --graph_dir=$GRAPH_DIR

GRAPH_DIR='./audioset_all_direct.pkl'

python3 utils/plot_statistics.py plot --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --select='1_full_train_gcn' --graph_dir=$GRAPH_DIR

python3 utils/plot_statistics.py plot --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --select='1_full_train_aug' --graph_dir=$GRAPH_DIR

GRAPH_DIR='./ATGCN_graph.pkl'

python3 utils/plot_statistics.py plot --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --select='1_full_train_gcn' --graph_dir=$GRAPH_DIR

python3 utils/plot_statistics.py plot --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --select='1_full_train_aug_gcn' --graph_dir=$GRAPH_DIR

