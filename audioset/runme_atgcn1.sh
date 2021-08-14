#!/bin/bash
#conda init
#conda activate py36
# Users can modify the following paths
DATASET_DIR="./audioset_tagging_cnn/raw_dataset"
WORKSPACE="./audioset_tagging_cnn"
GRAPH_DIR='./ATGCN_graph.pkl'

# ============ Train & Inference ============
MODEL_TYPE="Cnn14_16k_GCN"
python3 pytorch/main.py train --workspace=$WORKSPACE --graph_dir=$GRAPH_DIR --data_type='full_train' --window_size=512 --hop_size=160 --mel_bins=64 --fmin=50 --fmax=8000 --model_type=$MODEL_TYPE --loss_type='clip_bce' --balanced='balanced' --augmentation='mixup' --batch_size=64 --learning_rate=1e-3 --resume_iteration=0 --early_stop=1000000 --cuda
