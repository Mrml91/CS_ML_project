import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score

# HELPERS

def chunks_iterator(N, size): # with np.array convention 
    chunk_size = int(np.ceil(size / N))
    if chunk_size == 0:
        yield 0, size
    else:
        i = 0
        while i < size:
            yield i, i + chunk_size
            i += chunk_size
    

def print_bis(txt):
    print(txt, end='\x1b[1K\r')
    
def print_ter(txt):
    print(f"\n{txt}")

    
def make_timeline(freq):
    """
    ARGS:
        freq (int): frequency in Hertz
    
    RETURNS:
        (pd.timedelta_range) : timestamps for a signal sampled at <freq> Hz for 30 seconds
    """
    return pd.timedelta_range(start='0s', end='30s', periods=freq*30)


def make_full_timeline(windows, freq):
    # test there is no missing data
    deltas = np.unique(np.diff(windows))
    assert (len(deltas) == 1) and (int(deltas[0]) == 1)
    return pd.timedelta_range(start='0s',
                              end=pd.to_timedelta('30s') * (windows[-1] + 1),
                              periods=freq * 30 * (windows[-1] + 1))

def get_subject_ids(h5_file):
    return np.unique(h5_file["index"][:])

    
def get_subject_boundaries(h5_file, subject_id, ready_to_use=True):
    """
    Helper function to select data relating to a given subject (on numpy arrays)
    
    ARGS:
        h5_file (h5py.File)
        subject_id (int)
        ready_to_use (bool, default=True): return a slice or a tuple
        
    RETURNS:
        subject_boundaries : (slice) (index_start, index_end+1) if <ready_to_use>
                             (tuple) (index_start, index_end) if not <ready_to_use>
                        
    """
    sids = h5_file['index'][:]
    assert subject_id in sids
    
    start = np.argmax(sids == subject_id)
    end = len(sids) - 1 - np.argmax(sids[::-1] == subject_id)
    
    if ready_to_use:
        return slice(start, end + 1) # for numpy arrays
    return (start, end)


def get_subject_feature_signals(h5_file, subject_id, feature, frequencies_dict, as_timeseries=False):
    """
    Get the full timeseries for a given (subject_id, feature) pair.
    
    ARGS:
        h5_file (h5py.File)
        subject_id (int)
        feature (str)
        
    RETURNS:
        timeseries : (pd.Series if <as_timeseries>) represents the <feature> timeseries of the subject 
                     (list[np.array[?]] if not <as_timeseries>) list of <feature> signals from the subject
    """
    # Fetch subject boundaries
    boundaries = get_subject_boundaries(h5_file, subject_id)
    # Retrieve samples
    feature_timeseries = h5_file[feature][boundaries]
    if not as_timeseries:
        return feature_timeseries
    feature_timeseries = np.concatenate(feature_timeseries, axis=0)
    # Build timeline
    feature_frequency = frequencies_dict[feature]
    windows = h5_file['index_window'][boundaries]
    timeline = make_full_timeline(windows, feature_frequency)
    return pd.Series(data=feature_timeseries, index=timeline)


def get_subject_sleep_stage(subject_id, h5_train, y_train):
    start, end = get_subject_boundaries(h5_train, subject_id, ready_to_use=False)
    return y_train.loc[start:end] # because loc includes <end> (different behaviour than numpy arrays)
    

def subjects_ids_to_indexers(h5_file, subjects_ids, as_indices=False, as_boolean_array=False):
    if as_indices == as_boolean_array:
        raise NameError('Choose between `indices` and `boolean array` representations')
    if as_indices:
        boundaries = [get_subject_boundaries(h5_file, sid, ready_to_use=False) for sid in subjects_ids]
        return sum(map(lambda bounds: list(range(bounds[0], bounds[1]+1)), boundaries), list())
    if as_boolean_array:
        boolean_indexer = np.zeros(shape=(h5_file[list(h5_file.keys())[0]].shape[0],), dtype=bool)
        for sid in subjects_ids:
            boolean_indexer[get_subject_boundaries(h5_file, sid, ready_to_use=True)] = True
        return boolean_indexer
        
def get_eta_repr(elapsed, iteration, total_iterations):
    if iteration == 0:
        return "?"
    else:
        eta = (elapsed / iteration) * (total_iterations - iteration)
        return str(np.round(eta, 2)) + "s"


def custom_score(y_pred, y_true):
    return fbeta_score(y_pred=y_pred,
                       y_true=y_true,
                       labels=[0, 1, 2, 3, 4],
                       average="weighted",
                       beta=1)
