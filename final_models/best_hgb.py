import numpy as np
import pandas as pd

from helpers import subjects_ids_to_indexers, get_subject_ids, custom_score
from utils.distribution_statistics import make_input_new
from final_models.features_groups import *
from final_models.windows import concat_windows

from sklearn.experimental import enable_hist_gradient_boosting 
from sklearn.ensemble import HistGradientBoostingClassifier



def make_input_hgb_elem(h5_file):
    
    dfs = list()
    
    dfs.append( # df_bandlog
        make_input_new(
            h5_file,
            features=BAND_LOG_ENERGY_FEATURES_OLD,
            rescale_by_subject=False,
            moments=[1],
        )
    )
        
    dfs.append( # df_sleep = 
        make_input_new(
            h5_file,
            features=SLEEP_FEATURES[:2],
            rescale_by_subject=False,
            moments=[1]
        )
    )
        
    dfs.append( # df_logmod = 
        make_input_new(
            h5_file,
            features=OTHER_LOGMOD_FEATURES,
            rescale_by_subject=False,
            #interquantiles=[(0.2, 0.8)],
            quantiles_inv=[0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99],
            diff_orders=[0],
            interquantiles_inv=[(0.1, 0.9), (0.3, 0.7)],
        )
    )
    
 
    dfs.append( # df_time_diff_0 = 
        make_input_new(
            h5_file,
            features=sorted(set(TIME_FEATURES_OLD) - {"speed_norm"}),
            rescale_by_subject=False,
            hjorth=True, mmd=True,
            # hjorth=True,
            # moments=[1, 2],
            quantiles=[1e-4, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1-1e-4],
            # interquantiles=[(0.1, 0.9), (0.3, 0.7)],
            diff_orders=[0]
        )
    )
    
    dfs.append( # df_eeg_band_logmod = 
        make_input_new(
            h5_file,
            features=EEG_BAND_LOGMOD_FEATURES,
            rescale_by_subject=False,
            entropy=True, renyi_entropy=True,
            # hjorth=True,
            # moments=[1, 2],
            #quantiles=[1e-4, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1-1e-4],
            #interquantiles=[(0.1, 0.9), (0.3, 0.7)],
            diff_orders=[0],
            pre_op=lambda x: np.exp(2*x),
            pre_op_name='energy',
            post_op=np.log,
            post_op_name='log',
        )
    )
    
    res_df = pd.concat(dfs, axis=1, keys=[str(i) for i in range(len(dfs))])
    
    # Filling policy
    missing_values = res_df.isna().sum(axis=0)
    missing_values = missing_values.loc[missing_values > 0]
    if len(missing_values) > 0:
        print("Missing values :")
        print(missing_values)
        print("Filling missing values with zero")
        res_df = res_df.fillna(0)
        
    return res_df

# Gradient Boosting

class BestHGB:

    def __init__(self, h5_train, h5_test, y_train_arr, train_ids):
        self.model = HistGradientBoostingClassifier(verbose=1, random_state=1, max_iter=1000)
        
        self.shifts = [-1, 0, 1]
        self.h5_train = h5_train
        self.h5_test = h5_test
        
        self.train_ids = sorted(train_ids)
        self.train_ix = subjects_ids_to_indexers(self.h5_train, self.train_ids, as_indices=True)
        self.validation_ids = sorted(set(get_subject_ids(self.h5_train)) - set(self.train_ids))
        self.validation_ix = subjects_ids_to_indexers(self.h5_train, self.validation_ids, as_indices=True)       
        
        self.X_train_whole = self.make_input(h5_train)
        self.y_train_whole = y_train_arr
        
        self.X_train = self.X_train_whole.loc[self.train_ix, :]
        self.X_train = concat_windows(self.X_train, self.train_ids, self.h5_train, shifts=self.shifts)
        self.y_train = self.y_train_whole[self.train_ix]
        
        self.X_validation = self.X_train_whole.loc[self.validation_ix, :]
        self.X_validation = concat_windows(self.X_validation, self.validation_ids, self.h5_train, shifts=self.shifts)
        self.y_validation = self.y_train_whole[self.validation_ix]
        
        self.X_test = self.make_input(self.h5_test)
        test_ids = get_subject_ids(self.h5_test)
        self.X_test = concat_windows(self.X_test, test_ids, self.h5_test, shifts=self.shifts)

    def make_input(self, h5_file):
        return make_input_hgb_elem(h5_file)

    def train(self):
        self.model.fit(self.X_train, self.y_train)
            
    def predict_class(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X, policy):
        if policy == 'hard':
            return self.predict_class(X)
        else:
            return self.predict_proba(X)

    def predict_validation(self, policy):
        return self.predict(self.X_validation, policy=policy)

    def predict_test(self, policy):
        return self.predict(self.X_test, policy=policy)

    @property
    def validation_score(self):
        return custom_score(self.model.predict(self.X_validation), self.y_validation)


