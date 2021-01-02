# import sys
# sys.path.append('..')
# import numpy as np
# from helpers import *

# def _create_eeg_mean(h5_file, n_chunks=10, overwrite=False, verbose=True):
#     if (not overwrite) and ('eeg_mean_frontal' in h5_file.keys()) and ('eeg_mean_frontal_occipital' in h5_file.keys()):
#         return None
    
#     eegs_frontal = [f'eeg_{i}' for i in range(1, 4)]
#     eegs_frontal_occipital = [f'eeg_{i}' for i in range(4, 8)]
#     shape = h5_file["eeg_1"].shape
#     dtype = h5_file["eeg_1"].dtype
    
#     try:
#         h5_file.create_dataset('eeg_mean_frontal', shape=shape, dtype=dtype)
#         h5_file.create_dataset('eeg_mean_frontal_occipital', shape=shape, dtype=dtype)
#     except:
#         pass
    
#     for chunk_num, (chunk_start, chunk_end) in enumerate(chunks_iterator(n_chunks, shape[0])):
#         if verbose:
#             print_bis(f"{chunk_num+1}/{n_chunks}")
#         eeg_mean_frontal = sum( map(lambda eeg: h5_file[eeg][chunk_start:chunk_end], eegs_frontal) )
#         eeg_mean_frontal /= 3
        
#         eeg_mean_frontal_occipital = sum( map(lambda eeg: h5_file[eeg][chunk_start:chunk_end], eegs_frontal_occipital) )
#         eeg_mean_frontal_occipital /= 4
#         # store eeg_mean
#         h5_file["eeg_mean_frontal"][chunk_start:chunk_end] = eeg_mean_frontal
#         h5_file["eeg_mean_frontal_occipital"][chunk_start:chunk_end] = eeg_mean_frontal_occipital

#     return None