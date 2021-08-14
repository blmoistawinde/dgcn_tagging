#!/bin/bash
# You need to modify this path to your downloaded dataset directory
DATASET_DIR='./data'
FINE_GRAPH_DIR='./data/ontology_fid.pkl'
COARSE_GRAPH_DIR='./data/ontology_cid.pkl'

# You need to modify this path to your workspace to store features and models
WORKSPACE='./preprocessed'

# Hyper-parameters
GPU_ID=3
MODEL_TYPE='Cnn_9layers_AvgPooling_GCNEmb'
BATCH_SIZE=32

# Calculate feature
python utils/features.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --data_type='train' --workspace=$WORKSPACE
python utils/features.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --data_type='validate' --workspace=$WORKSPACE
python utils/features.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --data_type='evaluate' --workspace=$WORKSPACE

# Calculate scalar
python utils/features.py calculate_scalar --data_type='train' --workspace=$WORKSPACE


############ Train and validate on development dataset ############

# Train & inference

# Train
CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py train --dataset_dir=$DATASET_DIR --graph_dir $FINE_GRAPH_DIR --workspace=$WORKSPACE --taxonomy_level=fine --model_type=$MODEL_TYPE --holdout_fold=1 --batch_size=$BATCH_SIZE --cuda

# Inference and evaluate
# CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py inference_validation --dataset_dir=$DATASET_DIR --graph_dir $FINE_GRAPH_DIR --workspace=$WORKSPACE --taxonomy_level=fine --model_type=$MODEL_TYPE --holdout_fold=1 --iteration=2000 --batch_size=$BATCH_SIZE --cuda

CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py train --dataset_dir=$DATASET_DIR --graph_dir $COARSE_GRAPH_DIR --workspace=$WORKSPACE --taxonomy_level=coarse --model_type=$MODEL_TYPE --holdout_fold=1 --batch_size=$BATCH_SIZE --cuda

# CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py inference_validation --dataset_dir=$DATASET_DIR --graph_dir $COARSE_GRAPH_DIR --workspace=$WORKSPACE --taxonomy_level=coarse --model_type=$MODEL_TYPE --holdout_fold=1 --iteration=2000 --batch_size=$BATCH_SIZE --cuda


# Plot statistics
python utils/plot_results.py --workspace=$WORKSPACE --taxonomy_level=fine
python utils/plot_results.py --workspace=$WORKSPACE --taxonomy_level=coarse


############ Train on full development dataset and inference on evaluation dataset ############

# Train on full development dataset
# Train
# CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py train --dataset_dir=$DATASET_DIR --graph_dir $FINE_GRAPH_DIR --workspace=$WORKSPACE --taxonomy_level=fine --model_type=$MODEL_TYPE --holdout_fold=none --batch_size=$BATCH_SIZE --cuda

# Inference and evaluate
# CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py inference_validation --dataset_dir=$DATASET_DIR --graph_dir $FINE_GRAPH_DIR --workspace=$WORKSPACE --taxonomy_level=fine --model_type=$MODEL_TYPE --holdout_fold=none --iteration=3000 --batch_size=$BATCH_SIZE --cuda

# CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py train --dataset_dir=$DATASET_DIR --graph_dir $COARSE_GRAPH_DIR --workspace=$WORKSPACE --taxonomy_level=coarse --model_type=$MODEL_TYPE --holdout_fold=none --batch_size=$BATCH_SIZE --cuda

# CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py inference_validation --dataset_dir=$DATASET_DIR --graph_dir $COARSE_GRAPH_DIR --workspace=$WORKSPACE --taxonomy_level=coarse --model_type=$MODEL_TYPE --holdout_fold=none --iteration=3000 --batch_size=$BATCH_SIZE --cuda

############ END ############
