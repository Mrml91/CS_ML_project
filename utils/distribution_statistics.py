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
        

def get_distribution_moments(arr, moments=[]):
    res = np.empty(shape=(arr.shape[0], len(moments)))
    for k in range(len(moments)):
        res[:, k] = np.mean(arr ** moments[k], axis=1, keepdims=False)
    return res


def differentiate(signals, order, dropna=True):
    diff_signals = np.diff(signals, n=order, axis=1)
    if dropna:
        diff_signals = diff_signals[:, order:]
    return diff_signals


def _make_input_multidimensional_feature_chunk(
        sequences, quantiles=[], quantiles_inv=[], moments=[],
        diff_order=0, pre_op=do_nothing):
    """
    pre_op applied before differentiation
    """
    n_samples = sequences.shape[0]
    n_cols = len(quantiles) + len(quantiles_inv) + len(moments)
    assert n_cols > 0
    res = np.empty(shape=(n_samples, n_cols))
    diff_sequences = differentiate(pre_op(sequences), order=diff_order, dropna=True)
    if len(quantiles) > 0:
        res[:, :len(quantiles)] = get_distribution_quantiles(diff_sequences, quantiles)
    if len(quantiles_inv) > 0:
        res[:, len(quantiles):len(quantiles_inv) + len(quantiles)] =\
            get_distribution_quantiles_inv(diff_sequences, quantiles_inv)
    if len(moments) > 0:
        res[:, -len(moments):] = get_distribution_moments(diff_sequences, moments=moments)
    return res
        

def make_input_multidimensional_feature(
        h5_file, feature,
        quantiles=[], quantiles_inv=[], moments=[], diff_order=0, 
        pre_op=do_nothing, n_chunks=100):

    n_cols = len(quantiles) + len(quantiles_inv) + len(moments) 
    feature_array = np.empty(shape=(h5_file[feature].shape[0], n_cols))
    suffix = f"_diff_{diff_order}" if diff_order > 0 else ""
    columns = [(feature + suffix, f'qt_{q}') for q in quantiles] +\
              [(feature + suffix, f'qt_inv_{q_inv}') for q_inv in quantiles_inv] +\
              [(feature + suffix, f"moment_{mom}") for mom in moments]
    
    for i, j in chunks_iterator(n_chunks, h5_file[feature].shape[0]):
        feature_array[i:j, :] = \
            _make_input_multidimensional_feature_chunk(
                sequences=h5_file[feature][i:j],
                quantiles=quantiles,
                quantiles_inv=quantiles_inv,
                moments=moments,
                diff_order=diff_order,
                pre_op=pre_op)
        
    return feature_array, columns


def rescale_by_id(arr, h5_file):
    sids = get_subject_ids(h5_file)
    for sid in sids:
        indices = subjects_ids_to_indexers(h5_file, [sid], as_indices=True)
        arr[indices, :] -= np.mean(arr[indices, :], axis=0, keepdims=True)
        std = np.sqrt(np.mean(arr[indices, :] ** 2, axis=0, keepdims=True))
        arr[indices, :] /= np.where(std > 0, std, 1)
    return arr      


def make_input_new(
    h5_file, features=[], 
    quantiles=[], quantiles_inv=[], moments=[], diff_orders=[0],
    rescale_by_subject=True,
    pre_op=do_nothing, post_op=do_nothing, pre_op_name="", post_op_name=""):
    
    n_cols = len(quantiles) + len(quantiles_inv) + len(moments)
    input_arr = np.empty(shape=(h5_file["index"].shape[0], n_cols * len(diff_orders) * len(features)))
    i = 0
    columns = list()
    for cnt, feat in enumerate(features):
        print_bis(f"Feature #{cnt+1}/{len(features)}")
        for diff_order in diff_orders:
            input_arr[:, i:i+n_cols], cols = make_input_multidimensional_feature(
                h5_file, feature=feat, 
                quantiles=quantiles, quantiles_inv=quantiles_inv, moments=moments,
                diff_order=diff_order, pre_op=pre_op)
            columns = columns + cols
            i += n_cols
    if rescale_by_subject:
        input_arr = rescale_by_id(input_arr, h5_file)          
    df = pd.DataFrame(input_arr, columns=pd.MultiIndex.from_tuples(columns))
    df = post_op(df)
    cols = [(">>".join([pre_op_name, col[0], post_op_name]), *col[1:]) for col in df.columns]
    df.columns = pd.MultiIndex.from_tuples(cols)
    return df


