import numpy as np
import pandas as pd

from helpers import subjects_ids_to_indexers, get_subject_ids, custom_score
from utils.distribution_statistics import make_input_new
from final_models.features_groups import *
from final_models.windows import concat_windows

from final_models.best_rf import make_input_best_rf
from sklearn.ensemble import BaggingClassifier


class ClassicBG:

    def __init__(self, h5_train, h5_test, y_train_arr, train_ids):
        self.model = BaggingClassifier(verbose=1, random_state=1, n_estimators=70, n_jobs=-2)
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
        return make_input_best_rf(h5_file)

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


