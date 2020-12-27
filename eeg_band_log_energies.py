from scipy.fft import fft
import numpy as np
from helpers import *

# SIGNAL PROCESSING

# max frequency in our fourier transform: 25 Hz so no gamma
BANDS_FRONTIERS = [-1, 4, 8, 13, 22]
BANDS_LABELS = ['delta', 'theta', 'alpha', 'beta']

"""
def get_spectrum(seq, fs):
    ft_modulus = np.abs(fft(seq))
    # The signal is real so the spectrum is symmetric 
    if len(seq) % 2 == 0:
        ft_modulus = ft_modulus[:len(seq) // 2]
    else:
        ft_modulus = ft_modulus[:len(seq) // 2 + 1]
    freqs = np.arange(0, len(ft_modulus)) * fs / len(seq) # frequencies of the spectrum
    return pd.Series(data=ft_modulus, index=freqs)


def get_energy_by_band(seq, fs):
    spectrum = get_spectrum(seq, fs)
    bands = pd.cut(spectrum.index,
                   bins=BANDS_FRONTIERS,
                   labels=BANDS_LABELS
                  )
    energy = spectrum.pow(2).groupby(bands).sum() # energy proportional to this
    # energy.clip(1e-10, None)
    return energy
"""

def get_spectrum_energy_chunk(sequences, sampling_freq):
    fourier_transform = fft(sequences, axis=1)
    energy = np.power(fourier_transform.real, 2) + np.power(fourier_transform.imag, 2) # proportional to energy
    energy = energy[:, :int(np.ceil(sequences.shape[1] / 2))] # Shannon
    frequencies = np.arange(0, energy.shape[1]) * sampling_freq / sequences.shape[1]
    bands = pd.cut(frequencies, bins=BANDS_FRONTIERS, labels=BANDS_LABELS)
    energy_by_band = pd.DataFrame(data=energy, columns=bands)
    energy_by_band = energy_by_band.groupby(energy_by_band.columns, axis=1).sum()
    return energy_by_band
    
    

    

def _create_log_energy(h5_file, n_chunks=10, overwrite=False, verbose=True):
    if (not overwrite) and ('alpha_eeg_1_logE' in h5_file.keys()):
        return None
    
    eegs = [f'eeg_{i}' for i in range(1, 8)]
    shape = (h5_file["eeg_1"].shape[0], 1)
    dtype = h5_file["eeg_1"].dtype
    
    for band_name in BANDS_LABELS:
        for eeg in eegs:
            try:
                h5_file.create_dataset(f"{band_name}_{eeg}_logE", shape=shape, dtype=dtype)
            except:
                pass
    
    for chunk_num, (chunk_start, chunk_end) in enumerate(chunks_iterator(n_chunks, shape[0])):
        if verbose:
            print_bis(f"{chunk_num+1}/{n_chunks}")
        for eeg in eegs:
            energy = get_spectrum_energy_chunk(h5_file[eeg][chunk_start:chunk_end], 50)
            log_energy = np.log(1 + energy)
            for band_name in BANDS_LABELS:
                h5_file[f"{band_name}_{eeg}_logE"][chunk_start:chunk_end] = log_energy[[band_name]].values
    return None
    
        

#def get_spectrum_maxima(seq, fs, thresh=0.1):
#    spectrum = get_spectrum(seq, fs)
#    delta_left = np.diff(spectrum, prepend=spectrum[0] - 1) > 0 # ascending
#    delta_right = np.diff(spectrum[::-1], prepend=spectrum[-1] - 1)[::-1] > 0 # descending
#    ix_keep = np.logical_and(delta_left, delta_right) # local maximum
#    spectrum_util = spectrum.loc[ix_keep]
#    spectrum_util = spectrum_util.loc[spectrum_util > spectrum_util.max() * thresh]
#    return spectrum_util
