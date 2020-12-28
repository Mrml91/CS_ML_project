import sys
sys.path.append('..')
from scipy.fft import fft
import numpy as np
from helpers import *

def ceil_half(x):
    return np.ceil(x / 2).astype(int)

def get_spectrum_modulus(signals, sampling_freq):
    fourier_transform = fft(signals, axis=1)
    modulus = np.abs(fourier_transform)
    # Keep only the first part of the spectrum (Shannon)
    half_spectrum = ceil_half(signals.shape[1])
    modulus = modulus[:, :half_spectrum]
    return modulus


FEATURES_TO_CONVERT = [
    'accel_norm',
    'eeg_1',
    'eeg_2',
    'eeg_3',
    'eeg_4',
    'eeg_5',
    'eeg_6',
    'eeg_7',
    'pulse',
    'speed_norm',
]

def _create_log_modulus(
    h5_file, n_chunks=10, *,
    features_to_convert=FEATURES_TO_CONVERT, overwrite=False, verbose=True):
    
    # Check if the features already exist
    if (not overwrite) and all(map(lambda x: f"{x}_ft_logmod" in h5_file.keys(), features_to_convert)):
        return None
    
    features_frequencies = {feat: h5_file[feat][0].shape[0] // 30 for feat in features_to_convert}
    frequential_features = {feat: f"{feat}_ft_logmod" for feat in features_to_convert}
    for orig_feat, freq_feat in frequential_features.items():
        if freq_feat in h5_file.keys():
            continue
        else:
            n_samples, sample_size = h5_file[orig_feat].shape
            spectrum_size = ceil_half(sample_size)
            h5_file.create_dataset(
                freq_feat, 
                shape=(n_samples, spectrum_size),
                dtype=h5_file[orig_feat].dtype
            )
    n_samples, _ = h5_file[orig_feat].shape
            
    for chunk_num, (chunk_start, chunk_end) in enumerate(chunks_iterator(n_chunks, n_samples)):
        if verbose:
            print_bis(f"{chunk_num+1}/{n_chunks}")
        for orig_feat, freq_feat in frequential_features.items():
            sampling_freq = features_frequencies[orig_feat]
            h5_file[freq_feat][chunk_start : chunk_end] = log_spec = np.log(
                1 + get_spectrum_modulus(h5_file[orig_feat][chunk_start : chunk_end], sampling_freq)
            )
            
    return None
    