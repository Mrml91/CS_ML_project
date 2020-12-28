import numpy as np
import pandas as pd
import sys
sys.path.append("..")
from helpers import *
from utils.globals import *

def do_nothing(x):
    return x

def get_distribution_quantiles(arr, quantiles, **kwargs):
    return np.quantile(arr, q=quantiles, axis=1, **kwargs).T

def get_distribution_quantiles_inv(arr, quantiles, **kwargs):
    cdf = np.cumsum(arr, axis=1, **kwargs) 
    cdf /= cdf[:, [-1]]
    res = np.empty(shape=(arr.shape[0], len(quantiles)))
    for i, q in enumerate(quantiles):
        res[:, i] = np.argmax(q <= cdf, axis=1)
    return res
        

def get_distribution_characteristics(arr, truncate_dist=False, moments=[1, 2, 3, 4]):
    """
    mean, variance, skewness, kurtosis
    """
    if truncate_dist:
        inf = get_distribution_quantiles(arr, [0.005], keepdims=False)
        sup = get_distribution_quantiles(arr, [0.995], keepdims=False)
        return get_distribution_characteristics(np.clip(arr, inf, sup), truncate_dist=False)
    res = np.empty(shape=(arr.shape[0], len(moments)))
    for k in range(len(moments)):
        res[:, k] = np.mean(arr ** moments[k], axis=1, keepdims=False)
    return res


def differentiate(signals, order, dropna=True):
    diff_signals = np.diff(signals, n=order, axis=1)
    if dropna:
        diff_signals = diff_signals[order:, :]
    return np.diff(signals, n=order, axis=1)


def _make_input_multidimensional_feature_chunk(
        sequences, quantiles=QUANTILES, quantiles_inv=[], dist_char=True, truncate_dist=False, 
        diff_order=0, moments=[1, 2, 3, 4], pre_op=do_nothing):
    """
    pre_op applied before differentiation
    """
    n_samples = sequences.shape[0]
    n_cols = len(quantiles) + len(quantiles_inv) + len(moments) * int(dist_char)
    assert n_cols > 0
    res = np.empty(shape=(n_samples, n_cols))
    diff_sequences = differentiate(pre_op(sequences), order=diff_order, dropna=True)
    if len(quantiles) > 0:
        res[:, :len(quantiles)] = get_distribution_quantiles(diff_sequences, quantiles)
    if len(quantiles_inv) > 0:
        res[:, len(quantiles):len(quantiles_inv) + len(quantiles)] = get_distribution_quantiles_inv(diff_sequences, quantiles_inv)
    if dist_char:
        res[:, -len(moments):] = get_distribution_characteristics(diff_sequences, truncate_dist=truncate_dist, moments=moments)
    return res
        

def make_input_multidimensional_feature(
        h5_file, feature, quantiles=QUANTILES, quantiles_inv=[], dist_char=True, truncate_dist=False,
        n_chunks=100, diff_order=0, moments=[1, 2, 3, 4], pre_op=do_nothing):

    n_cols = len(quantiles) + len(quantiles_inv) + len(moments) * int(dist_char)
    feature_array = np.empty(shape=(h5_file[feature].shape[0], n_cols))
    suffix = f"_diff_{diff_order}" if diff_order > 0 else ""
    columns = [(feature + suffix, f'qt_{q}') for q in quantiles] +\
              [(feature + suffix, f'qt_inv_{q_inv}') for q_inv in quantiles_inv] +\
              [(feature, f"moment_{mom}") for mom in moments]
    
    for i, j in chunks_iterator(n_chunks, h5_file[feature].shape[0]):
        feature_array[i:j, :] = _make_input_multidimensional_feature_chunk(
            h5_file[feature][i:j], quantiles, quantiles_inv, dist_char, truncate_dist,
            diff_order=diff_order, moments=moments, pre_op=pre_op)
        
    return feature_array, columns


### Rescaling
from sklearn.preprocessing import StandardScaler 
# already robust on not logE features because we take quantiles
# --> StandardScaler 



def make_input(h5_file, features=FEATURES, quantiles=QUANTILES, 
               dist_char=True, truncate_dist=False, rescale=True,
               time_features=TIME_FEATURES, diff_orders=[0, 1, 2], 
               moments=[1, 2, 3, 4], pre_op=do_nothing, post_op=do_nothing):
    n_mono = sum([feat in MONO_FEATURES for feat in features])
    n_multi = len(features) - n_mono
    n_cols_mono = 1
    n_cols_multi = (len(quantiles) + len(moments) * int(dist_char)) * len(diff_orders)
    n_cols = n_mono * n_cols_mono + n_multi * n_cols_multi
    input_arr = np.empty(shape=(h5_file["index"].shape[0], n_cols))
    i = 0
    columns = list()
    for cnt, feat in enumerate(features):
        print_bis(f"Feature #{cnt+1}/{len(features)}")
        if feat in MONO_FEATURES:
            input_arr[:, [i]] = pre_op(h5_file[feat][:])
            columns = columns + [(feat, "")]
            i += 1
        else:
            for diff_order in diff_orders:
                input_arr[:, i:i+n_cols_multi], cols = make_input_multidimensional_feature(
                    h5_file, feat, quantiles, list(), dist_char, truncate_dist, 
                    diff_order=diff_order, pre_op=pre_op, moments=moments)
                columns = columns + cols
                i += n_cols_multi
    if rescale:
        ids = get_subject_ids(h5_file)
        for id in ids:
            indices = subjects_ids_to_indexers(h5_file, [id], as_indices=True, as_boolean_array=False)
            z_scaler = StandardScaler()
            input_arr[indices, :] = z_scaler.fit_transform(input_arr[indices, :])            
    df = pd.DataFrame(input_arr, columns=pd.MultiIndex.from_tuples(columns))
    df = post_op(df)
    return df


def rescale_by_id(arr, h5_file):
    sids = get_subject_ids(h5_file)
    scaler = StandardScaler()
    for sid in sids:
        indices = subjects_ids_to_indexers(h5_file, [sid], as_indices=True, as_boolean_array=False)
        arr[indices, :] = scaler.fit_transform(arr[indices, :]) 
    return arr      


def make_input_bis(h5_file, features=FEATURES, quantiles=[], quantiles_inv=[],
                truncate_dist=False, rescale=True,
                diff_orders=[0],  moments=[1, 2, 3, 4], 
                pre_op=do_nothing, post_op=do_nothing,
                pre_op_name="", post_op_name=""):
    dist_char = len(moments) > 0
    n_cols = len(quantiles) + len(quantiles_inv) + len(moments) * int(len(diff_orders) > 0)
    input_arr = np.empty(shape=(h5_file["index"].shape[0], n_cols * len(diff_orders) * len(features)))
    i = 0
    columns = list()
    for cnt, feat in enumerate(features):
        print_bis(f"Feature #{cnt+1}/{len(features)}")
        for diff_order in diff_orders:
            input_arr[:, i:i+n_cols], cols = make_input_multidimensional_feature(
                h5_file, feat, quantiles, quantiles_inv, dist_char, truncate_dist, 
                diff_order=diff_order, pre_op=pre_op, moments=moments)
            columns = columns + cols
            i += n_cols
    if rescale:
        input_arr = rescale_by_id(input_arr, h5_file)          
    df = pd.DataFrame(input_arr, columns=pd.MultiIndex.from_tuples(columns))
    df = post_op(df)
    cols = [(">>".join([pre_op_name, col[0], post_op_name]), *col[1:]) for col in df.columns]
    df.columns = pd.MultiIndex.from_tuples(cols)
    return df


