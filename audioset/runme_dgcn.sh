#!/bin/bash
# conda init
# conda activate py36
# Users can modify the following paths
DATASET_DIR="./audioset_tagging_cnn/raw_dataset"
WORKSPACE="./audioset_tagging_cnn"
GRAPH_DIR='./audioset_all_direct.pkl'
GRAPH_DIR2='./aser_all_pre_conj_0.pkl'

# ============ Train & Inference ============
MODEL_TYPE="Cnn14_16k_DoubleGCNInter"
python3 pytorch/main.py  train --workspace=$WORKSPACE --graph_dir=$GRAPH_DIR --graph_dir2=$GRAPH_DIR2 --data_type='full_train' --window_size=512 --hop_size=160 --mel_bins=64 --fmin=50 --fmax=8000 --model_type=$MODEL_TYPE --loss_type='clip_bce' --balanced='balanced' --augmentation='mixup' --batch_size=64 --learning_rate=3e-4 --resume_iteration=0 --early_stop=1000000 --cuda
