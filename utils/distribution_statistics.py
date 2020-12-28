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
    return res


def _make_input_multidimensional_feature_chunk(sequences, quantiles=QUANTILES, dist_char=True, truncate_dist=False):
    n_samples = sequences.shape[0]
    n_cols = len(quantiles) * int(len(quantiles) > 0) + 4 * int(dist_char)
    assert n_cols > 0
    res = np.empty(shape=(n_samples, n_cols))
    res[:, :len(quantiles)] = get_distribution_quantiles(sequences, quantiles)
    if dist_char:
        res[:, -4:] = get_distribution_characteristics(sequences, truncate_dist=truncate_dist)
    return res
        

def make_input_multidimensional_feature(h5_file, 
                                        feature, 
                                        quantiles=QUANTILES, 
                                        dist_char=True,
                                        truncate_dist=False,
                                        n_chunks=100):
    n_cols = len(quantiles) * int(len(quantiles) > 0) + 4 * int(dist_char)
    feature_array = np.empty(shape=(h5_file[feature].shape[0], n_cols))
    columns = [(feature, str(q)) for q in quantiles] + [(feature, f"Mom_{i}") for i in range(1,5) if dist_char]
    
    for i, j in chunks_iterator(n_chunks, h5_file[feature].shape[0]):
        feature_array[i:j, :] = _make_input_multidimensional_feature_chunk(
            h5_file[feature][i:j], quantiles, dist_char, truncate_dist)
        
    return feature_array, columns


def make_input(h5_file, features=FEATURES, quantiles=QUANTILES, dist_char=True, truncate_dist=False):
    n_mono = sum([feat in MONO_FEATURES for feat in features])
    n_cols_multi = len(quantiles) + 4 * int(dist_char)
    n_cols = n_mono + n_cols_multi * (len(features) - n_mono)
    input_arr = np.empty(shape=(h5_file["index"].shape[0], n_cols))
    i = 0
    columns = list()
    for cnt, feat in enumerate(features):
        print_bis(f"Feature #{cnt}/{len(features)}")
        if feat in MONO_FEATURES:
            input_arr[:, [i]] = h5_file[feat][:]
            columns = columns + [(feat, "")]
            i += 1
        else:
            input_arr[:, i:i+n_cols_multi], cols = make_input_multidimensional_feature(
                h5_file, feat, quantiles, dist_char, truncate_dist)
            columns = columns + cols
            i += n_cols_multi
    return pd.DataFrame(input_arr, columns=pd.MultiIndex.from_tuples(columns))
    


