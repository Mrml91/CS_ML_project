import numpy as np
from helpers import *


def _create_time_features(h5_file, overwrite=False, verbose=True):
    """
    adds `pulse_max_freq` and `pulse_max_logE`
    """
    time_features = ["sleep_time", "sleep_left", "sleep_time_relative"]

    if (not overwrite) and ("sleep_left" in h5_file.keys()):
        return None
    

    shape = (h5_file["eeg_1"].shape[0], 1)
    dtype = h5_file["eeg_1"].dtype
    
    for feat_name in time_features:
        if feat_name in h5_file.keys():
            continue
        h5_file.create_dataset(feat_name, shape=shape, dtype=dtype)
        
    subjects_ids = get_subject_ids(h5_file)
    for sid in subjects_ids:
        start, end = get_subject_boundaries(h5_file, sid, ready_to_use=False)
        indices = np.arange(start, end+1, dtype=float)[:, None]
        h5_file['sleep_time'][start:end+1] = indices - start
        h5_file['sleep_left'][start:end+1] = end - indices
        h5_file['sleep_time_relative'][start:end+1] = (indices - start) / (end - start)
        
    return None

