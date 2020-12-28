import numpy as np
import pandas as pd
import sys
sys.path.append("..")
from helpers import *
from utils.globals import *


def get_distribution_quantiles(arr, quantiles, **kwargs):
    return np.quantile(arr, q=quantiles, axis=1, **kwargs).T


def get_distribution_characteristics(arr, truncate_dist=False):
    """
    mean, variance, skewness, kurtosis
    """
    if truncate_dist:
        inf = get_distribution_quantiles(arr, [0.005], keepdims=False)
        sup = get_distribution_quantiles(arr, [0.995], keepdims=False)
        return get_distribution_characteristics(np.clip(arr, inf, sup), truncate_dist=False)
    res = np.empty(shape=(arr.shape[0], 4))
    res[:, 0] = np.mean(arr, axis=1, keepdims=False) # mean [order 1]
    res[:, 1] = np.mean((arr - res[:, [0]]) ** 2, axis=1, keepdims=False) # variance [order 2]
    z_var = ( arr - res[:, [0]] ) / res[:, [1]] 
    res[:, 2] = np.mean(z_var ** 3, axis=1, keepdims=False) # skewness [order 3]
    res[:, 3] = np.mean(z_var ** 4, axis=1, keepdims=False) # kurtosis [order 4]
    res[:, 4] = np.mean(1 / arr, axis=1, keepdims=False) # harmonical mean [order -1])
    return res

def differentiate(signals, order, dropna=True):
    diff_signals = np.diff(signals, n=order, axis=1)
    if dropna:
        diff_signals = diff_signals[order:, :]
    return np.diff(signals, n=order, axis=1)


def _make_input_multidimensional_feature_chunk(
        sequences, quantiles=QUANTILES, dist_char=True, truncate_dist=False, order=0):
    n_samples = sequences.shape[0]
    n_cols = len(quantiles) * int(len(quantiles) > 0) + 5 * int(dist_char)
    assert n_cols > 0
    res = np.empty(shape=(n_samples, n_cols))
    diff_sequences = differentiate(sequences, order=order, dropna=True)
    res[:, :len(quantiles)] = get_distribution_quantiles(diff_sequences, quantiles)
    if dist_char:
        res[:, -4:] = get_distribution_characteristics(diff_sequences, truncate_dist=truncate_dist)
    return res
        

def make_input_multidimensional_feature(h5_file, 
                                        feature, 
                                        quantiles=QUANTILES, 
                                        dist_char=True,
                                        truncate_dist=False,
                                        n_chunks=100,
                                        order=0):
    n_cols = len(quantiles) * int(len(quantiles) > 0) + 4 * int(dist_char)
    feature_array = np.empty(shape=(h5_file[feature].shape[0], n_cols))
    suffix = f"_diff_{order}" if order > 0 else ""
    columns = [(feature + suffix, str(q)) for q in quantiles] + [(feature, f"Mom_{i}") for i in [1, 2, 3, 4, -1] if dist_char]
    
    for i, j in chunks_iterator(n_chunks, h5_file[feature].shape[0]):
        feature_array[i:j, :] = _make_input_multidimensional_feature_chunk(
            h5_file[feature][i:j], quantiles, dist_char, truncate_dist, order=order)
        
    return feature_array, columns


### Rescaling
from sklearn.preprocessing import StandardScaler 
# already robust on not logE features because we take quantiles
# --> StandardScaler 

def make_input(h5_file, features=FEATURES, quantiles=QUANTILES, 
               dist_char=True, truncate_dist=False, rescale=True,
               time_features=TIME_FEATURES, orders=[0, 1, 2]):
    n_mono = sum([feat in MONO_FEATURES for feat in features])
    n_time = sum([feat in TIME_FEATURES for feat in features])
    n_multi = len(features) - n_time - n_mono
    n_cols_mono = 1
    n_cols_time = (len(quantiles) + 5 * int(dist_char)) * len(orders)
    n_cols_multi = len(quantiles) + 5 * int(dist_char)
    n_cols = n_mono * n_cols_mono + n_time * n_cols_time + n_multi * n_cols_multi
    input_arr = np.empty(shape=(h5_file["index"].shape[0], n_cols))
    i = 0
    columns = list()
    for cnt, feat in enumerate(features):
        print_bis(f"Feature #{cnt+1}/{len(features)}")
        if feat in MONO_FEATURES:
            input_arr[:, [i]] = h5_file[feat][:]
            columns = columns + [(feat, "")]
            i += 1
        elif feat in TIME_FEATURES:
            for order in orders:
                input_arr[:, i:i+n_cols_multi], cols = make_input_multidimensional_feature(
                    h5_file, feat, quantiles, dist_char, truncate_dist, order=order)
                columns = columns + cols
                i += n_cols_multi
        else:
            input_arr[:, i:i+n_cols_multi], cols = make_input_multidimensional_feature(
                h5_file, feat, quantiles, dist_char, truncate_dist)
            columns = columns + cols
            i += n_cols_multi
    if rescale:
        ids = get_subject_ids(h5_file)
        for id in ids:
            indices = subjects_ids_to_indexers(h5_file, [id], as_indices=True, as_boolean_array=False)
            z_scaler = StandardScaler()
            input_arr[indices,:] = z_scaler.fit_transform(input_arr[indices,:])            
    return pd.DataFrame(input_arr, columns=pd.MultiIndex.from_tuples(columns))


