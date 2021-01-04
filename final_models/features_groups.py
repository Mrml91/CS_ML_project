import itertools


eeg_nums = list(range(1, 8))
greek_letters = ["alpha", "beta", "delta", "theta"]

EEG_FEATURES = [f"eeg_{i}" for i in eeg_nums]
EEG_BAND_FEATURES = [f"{greek}_eeg_{i}" for greek, i in itertools.product(greek_letters, eeg_nums)]
OTHER_TIME_FEATURES = ["speed_norm", "accel_norm", "pulse"]

EEG_LOGMOD_FEATURES = [f"{eeg}_ft_logmod" for eeg in EEG_FEATURES]
EEG_BAND_LOGMOD_FEATURES = [f"{eeg_band}_ft_logmod" for eeg_band in EEG_BAND_FEATURES]
OTHER_LOGMOD_FEATURES = [f"{time_feat}_ft_logmod" for time_feat in OTHER_TIME_FEATURES]

SLEEP_FEATURES = ['sleep_left', 'sleep_time', 'sleep_time_relative']

# OLD NAMES
BAND_LOG_ENERGY_FEATURES_OLD = [f"{greek}_{eeg}_logE" for greek, eeg in itertools.product(greek_letters, EEG_FEATURES)]
LOGMOD_FEATURES_OLD = EEG_LOGMOD_FEATURES + OTHER_LOGMOD_FEATURES
TIME_FEATURES_OLD = EEG_FEATURES + OTHER_TIME_FEATURES