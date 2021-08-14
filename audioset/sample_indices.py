import os
import h5py
import numpy as np
from torch import chunk

# change to where the hdf5s stores
work_space = "./audioset_tagging_cnn"

index_path = f"{work_space}/hdf5s/indexes/full_train.h5"
hf = h5py.File(index_path, 'r')

indices = np.arange(len(hf["audio_name"]))
np.random.shuffle(indices)
print(len(hf["audio_name"]))
for portion in [1, 5, 10]:
    sel_nums = int(len(hf["audio_name"]) / 100 * portion)
    out_path = f"{work_space}/hdf5s/indexes/full_train_{portion}.h5"
    sel_indices = sorted(indices[:sel_nums])
    print(portion, sel_nums, len(sel_indices))
    with h5py.File(out_path, 'w') as hw:
        # indexing on hf directly is very slow, use value to convert to np array
        hw.create_dataset('audio_name', data=hf['audio_name'].value[sel_indices], dtype='S20')
        hw.create_dataset('target', data=hf['target'].value[sel_indices], dtype=np.bool)
        hw.create_dataset('hdf5_path', data=hf['hdf5_path'].value[sel_indices], dtype='S200')
        hw.create_dataset('index_in_hdf5', data=hf['index_in_hdf5'].value[sel_indices], dtype=np.int32)

