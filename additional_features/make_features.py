import h5py

from additional_features.eeg_band_log_energies import _create_log_energy
from additional_features.features_to_frequential import _create_log_modulus
from additional_features.pulse_to_freq import _create_pulse_max_log_energy_and_freq
from additional_features.speed_and_accel import _create_speed_and_acceleration
from additional_features.time_features import _create_time_features



def make_all_features(h5_train=None, h5_test=None, overwrite=False, verbose=True, n_chunks=100):

    train_file = "./kaggle_data/X_train.h5/X_train.h5"
    test_file = "./kaggle_data/X_test.h5/X_test.h5"
    h5_train = h5py.File(train_file, mode='a')
    h5_test = h5py.File(test_file, mode='a')

    for create_func in [_create_log_energy, _create_log_modulus, _create_pulse_max_log_energy_and_freq,
                        _create_speed_and_acceleration, _create_time_features]:
        for h5_file in [h5_train, h5_test]:
            create_func(h5_file, n_chunks=n_chunks, overwrite=overwrite, verbose=verbose)

    h5_train.close()
    h5_test.close()
