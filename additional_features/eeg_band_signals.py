import re
from scipy.signal import butter, lfilter
from helpers import *

BANDS_FRONTIERS = [-1, 4, 8, 13, 22]
BANDS_LABELS = ['delta', 'theta', 'alpha', 'beta']

BANDS = {
    'delta': [1e-2, 4],
    'theta': [4, 8],
    'alpha': [8, 13],
    'beta': [13, 22],
}

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


BANDS_BANDPASSES = {
    band: butter_bandpass(freqs[0], freqs[1], fs=50, order=5) 
    for band, freqs in BANDS.items()
}

# def butter_bandpass_filter(signals, lowcut, highcut, fs, order=5):
#     b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#     y = lfilter(b, a, signals, axis=1)
#     return y

def apply_filter(signals, b, a):
    return lfilter(b, a, signals, axis=1)

def get_band_signals(signals):
    return {band: apply_filter(signals, b, a) for band, (b, a) in BANDS_BANDPASSES.items()}


def _create_band_signals(h5_file, n_chunks=10, overwrite=False, verbose=True):
    if (not overwrite) and ('alpha_eeg_1' in h5_file.keys()):
        return None
    
    eegs = [feat for feat in h5_file.keys() if re.search("^eeg_\d$", feat)]
    shape = h5_file["eeg_1"].shape
    dtype = h5_file["eeg_1"].dtype
    
    for band_name in BANDS.keys():
        for eeg in eegs:
            try:
                h5_file.create_dataset(f"{band_name}_{eeg}", shape=shape, dtype=dtype)
            except:
                pass
    
    for chunk_num, (chunk_start, chunk_end) in enumerate(chunks_iterator(n_chunks, shape[0])):
        if verbose:
            print_bis(f"{chunk_num+1}/{n_chunks}")
        for eeg in eegs:
            band_signals = get_band_signals(h5_file[eeg][chunk_start:chunk_end])
            for band_name in BANDS.keys():
                h5_file[f"{band_name}_{eeg}"][chunk_start:chunk_end] = band_signals[band_name]
    
    return None
