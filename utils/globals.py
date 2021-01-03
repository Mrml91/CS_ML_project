import re
import h5py


def is_eeg_frontal(feature_name):
    return bool(re.search("eeg_mean_frontal", feature_name))


def is_band_time(feature_name):
    return bool(re.search("^(?:(?:alpha)|(?:beta)|(?:delta)|(?:theta))_eeg_\d$", feature_name))

with h5py.File("kaggle_data/X_train.h5/X_train.h5", mode='a') as h5_train:

    # Features
    IRRELEVANT_FEATURES = ['index', 'index_absolute', 'index_window',
                        'x', 'y', 'z',
                        'speed_x', 'speed_y', 'speed_z'
                        ]
    IRRELEVANT_FEATURES = IRRELEVANT_FEATURES + [feat for feat in h5_train.keys() if is_eeg_frontal(feat)]
    # print(IRRELEVANT_FEATURES)
    FEATURES = list(set(h5_train.keys()) - set(IRRELEVANT_FEATURES))

    TIME_FEATURES = ['accel_norm', 'eeg_1', 'eeg_2', 'eeg_3', 'eeg_4', 'eeg_5', 'eeg_6', 'eeg_7', 
                    #  'eeg_mean_frontal', 'eeg_mean_frontal_occipital', 
                    'pulse', 'speed_norm']
    
    BAND_FEATURES = [feat for feat in h5_train.keys() if is_band_time(feat)]
    
    SPECTRAL_FEATURES = [f"{time_feat}_ft_logmod" for time_feat in TIME_FEATURES]
    MONO_FEATURES = [feat for feat in FEATURES if h5_train[feat][0].shape[0] == 1]

    #print(set(MONO_FEATURES).union(set(TIME_FEATURES)).union(set(SPECTRAL_FEATURES)) -  set(FEATURES) )
    #print(set(FEATURES) - set(MONO_FEATURES).union(set(TIME_FEATURES)).union(set(SPECTRAL_FEATURES)) )
    #assert set(MONO_FEATURES).union(set(TIME_FEATURES)).union(set(SPECTRAL_FEATURES)) ==  set(FEATURES) 

    # Quantiles 

    LOWER_TAIL_QUANTILES = [0.01, 0.025, 0.05]
    DECILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    UPPER_TAIL_QUANTILES = [0.95, 0.975, 0.99]

    QUANTILES = LOWER_TAIL_QUANTILES + DECILES + UPPER_TAIL_QUANTILES
    TAIL_QUANTILES = LOWER_TAIL_QUANTILES + UPPER_TAIL_QUANTILES

