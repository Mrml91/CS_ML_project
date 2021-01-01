import sys
sys.path.append('..')
import numpy as np
from helpers import *
from additional_features.eeg_band_log_energies import get_spectrum_energy_chunk


def _create_eeg_mean(h5_file, n_chunks=10, overwrite=False, verbose=True):
    if (not overwrite) and ('eeg_mean' in h5_file.keys()):
        return None
    
    eegs = [f'eeg_{i}' for i in range(1, 8)]
    shape = h5_file["eeg_1"].shape
    dtype = h5_file["eeg_1"].dtype
    
    try:
        h5_file.create_dataset(f"eeg_mean", shape=shape, dtype=dtype)
    except:
        pass
    
    for chunk_num, (chunk_start, chunk_end) in enumerate(chunks_iterator(n_chunks, shape[0])):
        if verbose:
            print_bis(f"{chunk_num+1}/{n_chunks}")
        eeg_mean = sum( map(lambda eeg: h5_file[eeg][chunk_start:chunk_end], eegs) )
        eeg_mean /= 7
        # store eeg_mean
        h5_file[f"eeg_mean"][chunk_start:chunk_end] = eeg_mean

    return None
    
        

#def get_spectrum_maxima(seq, fs, thresh=0.1):
#    spectrum = get_spectrum(seq, fs)
#    delta_left = np.diff(spectrum, prepend=spectrum[0] - 1) > 0 # ascending
#    delta_right = np.diff(spectrum[::-1], prepend=spectrum[-1] - 1)[::-1] > 0 # descending
#    ix_keep = np.logical_and(delta_left, delta_right) # local maximum
#    spectrum_util = spectrum.loc[ix_keep]
#    spectrum_util = spectrum_util.loc[spectrum_util > spectrum_util.max() * thresh]
#    return spectrum_util
