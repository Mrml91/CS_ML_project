import numpy as np
import time
import sys
sys.path.append("..")
from helpers import *
import matplotlib.pyplot as plt
import seaborn as sns


# <input_maker>(h5_file, **params) --> X_<file>
# <input_shaper>(X_train, scaler, pca): --> func(X) --> X_shaped

# Multiple cross-validation
from itertools import combinations

def get_complement(a, omega):
    return sorted(set(omega) - set(a))

def get_n_train_validation_splits(n, train_size, subjects_ids, seed=None):
    """
    [<split_1>, <split_2>, ..., <split_n>]
    <split> = ([train_id_1, train_id_2, ..., train_id_<train_size>], [val_id_1, ...])
    """
    if seed:
        np.random.seed(seed)      
    all_train_combs = combinations(subjects_ids, train_size)
    train_combs = np.random.permutation(list(all_train_combs))[:n]
    train_combs = train_combs.tolist()
    splits = [(tc, get_complement(tc, subjects_ids)) for tc in train_combs]
    return splits


class InputMaker: 
    
    def __init__(self, maker_func):
        self.maker = maker_func
        
    def get_input(self, h5_file, seed=None):
        np.random.seed(seed)
        return self.maker(h5_file)
    

class InputShaper:
    
    def __init__(self, *operators, seed=None):
        assert all(op.copy for op in operators)
        self.operators = operators
        if seed is not None:
            self.set_seed(seed)
        
    def set_seed(self, seed):
        self.seed = seed
        for i in range(len(self.operators)):
            self.operators[i].random_state = self.seed
        
        
    def fit(self, X):
        for op in self.operators:
            X = op.fit_transform(X)
            
    def fit_transform(self, X):
        for op in self.operators:
            X = op.fit_transform(X)
        return X
        
    def transform(self, X):
        for op in self.operators:
            X = op.transform(X)
        return X
    

class PoolModels:
    
    def __init__(self, input_maker, n_splits, train_size, input_shaper, blueprint, parameters_list, 
                 h5_train=None, h5_test=None, X_train=None, X_test=None, y_train_arr=None, seed=None,
                 warming_params=dict(), convergence_params=dict()):
        # seed only works for the splits and the blueprint
        # should work for input_maker if only numpy operations
        # works for input_shaper
        self.seed = seed
        self.input_maker = input_maker
        self.n_splits = n_splits
        self.train_size = train_size
        self.input_shaper = input_shaper
        self.input_shaper.set_seed(self.seed) # works (?)
        self.blueprint = blueprint
        self.parameters_list = parameters_list
        # self.iterations_to_warm = iterations_to_warm
        self.warming_params = warming_params
        self.convergence_params = convergence_params

        self.h5_train = h5_train
        self.h5_test = h5_test
        
        self.splits = get_n_train_validation_splits(
            self.n_splits, 
            self.train_size, 
            subjects_ids=get_subject_ids(self.h5_train),
            seed=self.seed)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train_arr = y_train_arr
        
        # row = model, col = train-val-split
        self.models = np.zeros(shape=(len(self.parameters_list), self.n_splits), dtype='object')
        self.train_scores = np.zeros(shape=(len(self.parameters_list), self.n_splits), dtype=float)
        self.validation_scores = np.zeros(shape=(len(self.parameters_list), self.n_splits), dtype=float)
        # self.has_converged = np.zeros(shape=(len(self.parameters_list), self.n_splits), dtype=bool)
        
    def make_train_input(self):
        if (self.X_train is not None):
            return
        self.X_train = self.input_maker.get_input(self.h5_train, seed=self.seed)
    
    def make_test_input(self):
        if (self.X_test is not None):
            return
        self.X_test = self.input_maker.get_input(self.h5_test, seed=self.seed)
        
        
    def train_on_split(self, split_num, models_ix=None, training_params=dict(), step_name="TRAINING", score_train=False):
        self.make_train_input()

        train_selector = subjects_ids_to_indexers(self.h5_train, self.splits[split_num][0], as_boolean_array=True)
        val_selector = ~train_selector
        
        y_train_train, y_train_val = self.y_train_arr[train_selector], self.y_train_arr[val_selector]
        X_train_train, X_train_val = self.X_train[train_selector], self.X_train[val_selector]
        X_train_train = self.input_shaper.fit_transform(X_train_train)
        X_train_val = self.input_shaper.transform(X_train_val)

        start_time = time.time()
        total = len(self.parameters_list) if models_ix is None else len(models_ix)
        k = 0
        for i, params_set in enumerate(self.parameters_list):
            if (models_ix is not None) and (i not in models_ix):
                continue
            # eta
            eta = get_eta_repr(time.time() - start_time, k, total)
            print_bis(f"Split #{split_num+1}/{self.n_splits} - {step_name} Model #{k+1}/{total} [ETA: {eta}]")
            
            params_model = {k: v for k, v in params_set.items()}
            params_model.update(training_params)
            try:
                model = self.blueprint(random_state=self.seed, **params_model)
            except TypeError:
                model = self.blueprint(**params_model)
            model.fit(X_train_train, y_train_train)
            self.models[i, split_num] = model
            if score_train:
                self.train_scores[i, split_num] = custom_score(model.predict(X_train_train), y_train_train)
            self.validation_scores[i, split_num] = custom_score(model.predict(X_train_val), y_train_val)
            # if max_iter == -1:
            #     self.has_converged[i, split_num] = True
            
            k += 1
    
    def train_on_all_data(self, model):
        X_train_shaped = self.input_shaper.fit_transform(self.X_train)
        model.fit(X_train_shaped,self.y_train_arr)
        return model

    def _plot(self, scores):
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(scores, vmin=0, vmax=1, annot=True, ax=ax)
        plt.show()
    
    def plot_validation(self):
        self._plot(self.validation_scores)

    def plot_training(self):
        self._plot(self.train_scores)

    def warm_up(self):
        for split_num in range(self.n_splits):
            self.train_on_split(split_num, models_ix=None,
                                step_name='WARM UP',
                                score_train=False,
                                training_params=self.warming_params
                               )
    
    def select_n_best_models(self, n):
        average_validation_score = np.mean(self.validation_scores, axis=1)
        best_models_ix = np.argsort(average_validation_score)[-n:]
        return best_models_ix
    
    def train_n_best_models_until_convergence(self, n, split_num=0):        
        best_models_ix = self.select_n_best_models(n)
        self.train_on_split(split_num, training_params=self.convergence_params, models_ix=best_models_ix, step_name="TRAINING UNTIL CONVERGENCE", score_train=True)
        results = []
        for bm_ix in best_models_ix:
            results.append({"model": self.models[bm_ix, split_num], 
                            "train_score": self.train_scores[bm_ix, split_num],
                            "validation_score": self.validation_scores[bm_ix, split_num]
                           })
        return results

   