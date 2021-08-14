DATASET_DIR="./raw_dataset"
WORKSPACE="."

seq -w 00 40 | parallel --result outdir python3 utils/dataset.py pack_waveforms_to_hdf5 --csv_path=$DATASET_DIR"/metadata/unbalanced_partial_csvs/unbalanced_train_segments_part"{}".csv" --audios_dir=$DATASET_DIR"/audios/unbalanced_train_segments/unbalanced_train_segments_part"{} --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/unbalanced_train/unbalanced_train_part"{}".h5"
