import pandas as pd
import numpy as np

def shift_and_fill(df, shift):
    shifted_df = df.shift(shift)
    if shift > 0:
        shifted_df.bfill(inplace=True)
    elif shift < 0:
        shifted_df.ffill(inplace=True)
    return shifted_df


def roll_and_concat(df, shifts_range):
    return pd.concat(map(lambda shift: shift_and_fill(df, shift), shifts_range), 
                     axis=1, keys=shifts_range)    
    
def subjects_ids_col(h5_file):
    return h5_file["index"][:]

def concat_windows(arr, subjects_ids, h5_file, shifts): # subjects_ids must be sorted
    sid_col = subjects_ids_col(h5_file)
    sid_col = sid_col[np.isin(sid_col, subjects_ids)]
    df = pd.DataFrame(arr)
    
    return df.groupby(sid_col).apply(roll_and_concat, shifts_range=shifts)


def make_input_rolling(h5_file, shifts, make_input_function):
    df = make_input_function(h5_file)
    df_with_window = concat_windows(h5_file, df, shifts)
    return df_with_window
