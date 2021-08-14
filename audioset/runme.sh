#!/bin/bash
conda init
conda activate torch
# Users can modify the following paths
DATASET_DIR="./audioset_tagging_cnn/raw_dataset"
WORKSPACE="./audioset_tagging_cnn"

# ============ Train & Inference ============
MODEL_TYPE="Cnn14_16k"
python3 pytorch/main.py train --workspace=$WORKSPACE --data_type='balanced_train' --window_size=512 --hop_size=160 --mel_bins=64 --fmin=50 --fmax=8000 --model_type=$MODEL_TYPE --loss_type='clip_bce' --balanced='balanced' --augmentation='mixup' --batch_size=64 --learning_rate=1e-3 --resume_iteration=0 --early_stop=1000000 --cuda

## Plot statistics
python3 utils/plot_statistics.py plot --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --select=1_aug
