import h5py

with h5py.File("kaggle_data/X_train.h5/X_train.h5", mode='a') as h5_train:

    # Features
    IRRELEVANT_FEATURES = ['index', 'index_absolute', 'index_window',
                        'x', 'y', 'z',
                        'speed_x', 'speed_y', 'speed_z',
                        ]
    FEATURES = list(set(h5_train.keys()) - set(IRRELEVANT_FEATURES))


    TIME_FEATURES = ['accel_norm', 'eeg_1', 'eeg_2', 'eeg_3', 'eeg_4', 'eeg_5', 'eeg_6', 'eeg_7', 'pulse', 'speed_norm']
    SPECTRAL_FEATURES = [f"{time_feat}_ft_logmod" for time_feat in TIME_FEATURES]
    MONO_FEATURES = [feat for feat in FEATURES if h5_train[feat][0].shape[0] == 1]
    
    
    assert set(MONO_FEATURES).union(set(TIME_FEATURES)).union(set(SPECTRAL_FEATURES)) ==  set(FEATURES) 

    # Quantiles 

    LOWER_TAIL_QUANTILES = [0.01, 0.025, 0.05]
    DECILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    UPPER_TAIL_QUANTILES = [0.95, 0.975, 0.99]

    QUANTILES = LOWER_TAIL_QUANTILES + DECILES + UPPER_TAIL_QUANTILES
    TAIL_QUANTILES = LOWER_TAIL_QUANTILES + UPPER_TAIL_QUANTILES

