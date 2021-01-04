import numpy as np
import pandas as pd
import sys
sys.path.append("..")
from helpers import *
from utils.globals import *

def do_nothing(x):
    return x

def get_distribution_quantiles(arr, quantiles):
    return np.quantile(arr, q=quantiles, axis=1).T

def get_distribution_quantiles_inv(arr, quantiles_inv):
    cdf = np.cumsum(arr, axis=1) 
    cdf /= cdf[:, [-1]]
    res = np.empty(shape=(arr.shape[0], len(quantiles_inv)))
    for i, q in enumerate(quantiles_inv):
        res[:, i] = np.argmax(q <= cdf, axis=1)
    return res
        
def get_distribution_interquantiles(arr, interquantiles):
    # interquantiles = [(0.1, 0.9), (0.2, 0.8), ...]
    inf_q, sup_q = [x[0] for x in interquantiles], [x[1] for x in interquantiles]
    interquantiles_vals = \
        get_distribution_quantiles(arr, sup_q) - get_distribution_quantiles(arr, inf_q)
    return interquantiles_vals
    
def get_distribution_interquantiles_inv(arr, interquantiles_inv):
    # interquantiles_inv = [(0.1, 0.9), (0.2, 0.8), ...]
    inf_qinv, sup_qinv = [x[0] for x in interquantiles_inv], [x[1] for x in interquantiles_inv]
    interquantiles_inv_vals = \
        get_distribution_quantiles_inv(arr, sup_qinv) - get_distribution_quantiles_inv(arr, inf_qinv)
    return interquantiles_inv_vals


def get_distribution_moments(arr, moments=[]):
    res = np.empty(shape=(arr.shape[0], len(moments)))
    for k in range(len(moments)):
        res[:, k] = np.mean(arr ** moments[k], axis=1, keepdims=False)
    return res

def get_entropy(psd_signal): #https://dsp.stackexchange.com/questions/23689/what-is-spectral-entropy
    normalized_psd = psd_signal / np.sum(psd_signal, axis=1, keepdims=True)
    spectral_entropy = - np.sum(normalized_psd * np.log(normalized_psd), axis=1, keepdims=True)
    return spectral_entropy

def get_renyi_entropy(psd_signal):
    normalized_psd = psd_signal / np.sum(psd_signal, axis=1, keepdims=True)
    spectral_entropy = - np.log(np.sum(normalized_psd ** 2, axis=1, keepdims=True))
    return spectral_entropy

def get_hjorth_parameters(signals):#https://en.wikipedia.org/wiki/Hjorth_parameters
    # Activity | Mobility | Complexity
    res = np.empty(shape=(signals.shape[0], 3))
    v0, v1, v2 = [np.var(differentiate(signals, order, dropna=True), axis=1, keepdims=False)
                    for order in [0, 1, 2]]
    res[:, 0] = v0
    res[:, 1] = np.sqrt(v1 / v0)
    res[:, 2] = np.sqrt(v2 / v1) / res[:, 1]
    return res


# wavelength (lambda in the article) = 100 

def get_mmd_elem(subsignals):
    argmax_ = np.argmax(subsignals, axis=1)
    argmin_ = np.argmin(subsignals, axis=1)
    all_rows = list(range(subsignals.shape[0])) 
    dy = subsignals[all_rows, argmax_] - subsignals[all_rows, argmin_] # zip-like numpy indexing
    dx = (argmax_ - argmin_)
    mmd =  np.sqrt(dx**2 + dy**2)
    return mmd.reshape((mmd.shape[0], 1))

def get_mmd(signals, wavelength=100): # minimum maximum distance
    subsignals = np.split(signals, wavelength, axis=1) # not flex
    mmd_elems = map(get_mmd_elem, subsignals)
    mmd_total = sum(mmd_elems)
    return mmd_total


# TODO
# def esis(signals, f_mid=???):
    


def differentiate(signals, order, dropna=True):
    if order == 0:
        return signals
    if order < 0:
        diff_signals = np.copy(signals)
        for _ in range(-order):
            diff_signals = np.cumsum(diff_signals, axis=1)
    else:
        diff_signals = np.diff(signals, n=order, axis=1)
    if dropna:
        diff_signals = diff_signals[:, max(0, order):]
    return diff_signals


def _make_input_multidimensional_feature_chunk(
        sequences, 
        quantiles=[], quantiles_inv=[],
        moments=[],
        interquantiles=[], interquantiles_inv=[],
        entropy=False, renyi_entropy=False, hjorth=False, mmd=False,
        diff_order=0, pre_op=do_nothing):
    """
    pre_op applied before differentiation
    """
    n_samples = sequences.shape[0]
    n_cols =  len(quantiles) + len(quantiles_inv) \
            + len(moments) \
            + len(interquantiles) + len(interquantiles_inv) \
            + int(entropy) \
            + int(renyi_entropy) \
            + 3 * int(hjorth) \
            + int(mmd)
    assert n_cols > 0
    res = np.empty(shape=(n_samples, n_cols))
    diff_sequences = differentiate(pre_op(sequences), order=diff_order, dropna=True)
    ix = 0
    if len(quantiles) > 0:
        res[:, ix:ix+len(quantiles)] = get_distribution_quantiles(
            diff_sequences, quantiles)
        ix += len(quantiles)
    if len(quantiles_inv) > 0:
        res[:, ix:ix + len(quantiles_inv)] = get_distribution_quantiles_inv(
            diff_sequences, quantiles_inv)
        ix += len(quantiles_inv)
    if len(moments) > 0:
        res[:, ix:ix+len(moments)] = get_distribution_moments(
            diff_sequences, moments)
        ix += len(moments)
    if len(interquantiles) > 0:
        res[:, ix:ix+len(interquantiles)] = get_distribution_interquantiles(
            diff_sequences, interquantiles)
        ix += len(interquantiles)
    if len(interquantiles_inv) > 0:
        res[:, ix:ix+len(interquantiles_inv)] = get_distribution_interquantiles_inv(
            diff_sequences, interquantiles_inv)
        ix += len(interquantiles_inv)
    if entropy:
        res[:, [ix]] = get_entropy(diff_sequences)
        ix += 1
    if renyi_entropy:
        res[:, [ix]] = get_renyi_entropy(diff_sequences)
        ix += 1
    if hjorth:
        res[:, ix:ix+3] = get_hjorth_parameters(diff_sequences)
        ix += 3
    if mmd:
        res[:, [ix]] = get_mmd(diff_sequences)
        ix += 1
    return res
        

def make_input_multidimensional_feature(
        h5_file, feature,
        quantiles=[], quantiles_inv=[],
        moments=[], 
        interquantiles=[], interquantiles_inv=[],
        entropy=False, renyi_entropy=False, hjorth=False, mmd=False,
        diff_order=0, 
        pre_op=do_nothing, n_chunks=10):

    n_cols =  len(quantiles) + len(quantiles_inv) \
            + len(moments) \
            + len(interquantiles) + len(interquantiles_inv) \
            + int(entropy) \
            + int(renyi_entropy) \
            + 3 * int(hjorth) \
            + int(mmd)
    feature_array = np.empty(shape=(h5_file[feature].shape[0], n_cols))
    suffix = f"_diff_{diff_order}" if diff_order != 0 else ""
    columns = [(feature + suffix, f'qt_{q}') for q in quantiles] +\
              [(feature + suffix, f'qt_inv_{q_inv}') for q_inv in quantiles_inv] +\
              [(feature + suffix, f"moment_{mom}") for mom in moments] +\
              [(feature + suffix, f'interqt_{inf_iq}-{sup_iq}') for inf_iq, sup_iq in interquantiles] +\
              [(feature + suffix, f'interqt_inv_{inf_iq_inv}-{sup_iq_inv}') for inf_iq_inv, sup_iq_inv in interquantiles_inv] +\
              [(feature + suffix, 'entropy')] * int(entropy) +\
              [(feature + suffix, 'renyi_entropy')] * int(renyi_entropy) +\
              [(feature + suffix, f'Hjorth_{param}') for param in ("activity", "mobility", "complexity") if hjorth] +\
              [(feature + suffix, 'mmd')] * int(mmd)
    
    for i, j in chunks_iterator(n_chunks, h5_file[feature].shape[0]):
        feature_array[i:j, :] = \
            _make_input_multidimensional_feature_chunk(
                sequences=h5_file[feature][i:j],
                quantiles=quantiles,
                quantiles_inv=quantiles_inv,
                moments=moments,
                interquantiles=interquantiles,
                interquantiles_inv=interquantiles_inv,
                entropy=entropy, renyi_entropy=renyi_entropy, hjorth=hjorth, mmd=mmd,
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
    quantiles=[], quantiles_inv=[], 
    moments=[],
    interquantiles=[], interquantiles_inv=[],
    entropy=False, renyi_entropy=False, hjorth=False, mmd=False,
    diff_orders=[0],
    rescale_by_subject=True,
    pre_op=do_nothing, post_op=do_nothing, pre_op_name="", post_op_name=""):
    
    n_cols =  len(quantiles) + len(quantiles_inv) \
            + len(moments) + len(interquantiles) \
            + len(interquantiles_inv) \
            + int(entropy) \
            + int(renyi_entropy) \
            + 3 * int(hjorth)\
            + int(mmd)
    input_arr = np.empty(shape=(h5_file["index"].shape[0], n_cols * len(diff_orders) * len(features)))
    i = 0
    columns = list()
    for cnt, feat in enumerate(features):
        print_bis(f"Feature #{cnt+1}/{len(features)}")
        for diff_order in diff_orders:
            input_arr[:, i:i+n_cols], cols = make_input_multidimensional_feature(
                h5_file, feature=feat, 
                quantiles=quantiles, quantiles_inv=quantiles_inv,
                moments=moments,
                interquantiles=interquantiles, interquantiles_inv=interquantiles_inv,
                entropy=entropy, renyi_entropy=renyi_entropy, hjorth=hjorth, mmd=mmd,
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


