import numpy as np
from scipy.fft import fft
from helpers import *

# We only keep the frequency with the highest amplitude and its amplitude
PULSE_FREQ = 10

def get_energies(signals):
    fourier = fft(signals, axis=1)
    fourier = fourier[:, :int(np.ceil(fourier.shape[1] / 2))] # Shannon
    energies = fourier.real ** 2 + fourier.imag ** 2
    return energies

def get_max_energy_and_freq(signals, per_minute, sampling_freq):
    energies = get_energies(signals)
    argmax_frequencies = np.argmax(energies, axis=1)
    max_energies = energies[list(range(energies.shape[0])), argmax_frequencies]
    argmax_frequencies = argmax_frequencies * sampling_freq * 60 / signals.shape[1]
    return max_energies, argmax_frequencies



def _create_pulse_max_log_energy_and_freq(h5_file, n_chunks=10, overwrite=False, verbose=True):
    """
    adds `pulse_max_freq` and `pulse_max_logE`
    """

    if (not overwrite) and ("pulse_max_freq" in h5_file.keys()):
        return None
    
    shape = (h5_file["pulse"].shape[0], 1)
    dtype = h5_file["pulse"].dtype
    
    for feat_name in ('pulse_max_freq', 'pulse_max_logE'):
        if feat_name in h5_file.keys():
            continue
        h5_file.create_dataset(feat_name, shape=shape, dtype=dtype)
        
    for chunk_num, (chunk_start, chunk_end) in enumerate(chunks_iterator(n_chunks, shape[0])):
        if verbose:
            print_bis(f"{chunk_num+1}/{n_chunks}")
        max_energies, argmax_freqs = get_max_energy_and_freq(
            h5_file["pulse"][chunk_start:chunk_end], per_minute=True, sampling_freq=PULSE_FREQ
        )
        h5_file["pulse_max_logE"][chunk_start:chunk_end] = np.log(1 + max_energies)[:, None]
        h5_file["pulse_max_freq"][chunk_start:chunk_end] = argmax_freqs[:, None]

        
    return None

