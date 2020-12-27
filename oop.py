# <input_maker>(h5_file, **params) --> X_<file>
# <input_shaper>(X_train, scaler, pca): --> func(X) --> X_shaped

# Multiple cross-validation
from itertools import combinations

def get_complement(a, omega):
    return sorted(set(omega) - set(a))

def get_n_train_validation_splits(n, train_size, subjects_ids=get_subject_ids(h5_train), seed=None):
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

get_n_train_validation_splits(n=3, train_size=5, subjects_ids=[1,2,3,4,5,6,7], seed=2)


# from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class InputMaker: 
    
    def __init__(self, maker_func):
        self.maker = maker_func
        
    def get_train_input(self, seed=None):
        np.random.seed(seed)
        return self.maker(h5_train)
    
    def get_test_input(self, seed=None):
        np.random.seed(seed)
        return self.maker(h5_test)
    

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
                 iterations_to_warm=10, seed=None, X_train=None, X_test=None):
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
        self.iterations_to_warm = iterations_to_warm
        
        self.splits = get_n_train_validation_splits(self.n_splits, self.train_size, seed=self.seed)
        
        self.X_train = X_train
        self.X_test = X_test
        
        # row = model, col = train-val-split
        self.models = np.zeros(shape=(len(self.parameters_list), self.n_splits), dtype='object')
        self.train_scores = np.zeros(shape=(len(self.parameters_list), self.n_splits), dtype=float)
        self.validation_scores = np.zeros(shape=(len(self.parameters_list), self.n_splits), dtype=float)
        self.has_converged = np.zeros(shape=(len(self.parameters_list), self.n_splits), dtype=bool)
        
    def make_train_input(self):
        if (self.X_train is not None):
            return
        self.X_train = self.input_maker.get_train_input(seed=self.seed)
    
    def make_test_input(self):
        if (self.X_test is not None):
            return
        self.X_test = self.input_maker.get_test_input(seed=self.seed)
        
        
    def train_on_split(self, split_num, models_ix=None, max_iter=-1, step_name="TRAINING", score_train=False):
        self.make_train_input()
        
        train_selector = subjects_ids_to_indexers(h5_train, self.splits[split_num][0], as_boolean_array=True)
        val_selector = ~train_selector
        
        y_train_train, y_train_val = y_train_arr[train_selector], y_train_arr[val_selector]
        X_train_train, X_train_val = self.X_train[train_selector], self.X_train[val_selector]
        X_train_train = self.input_shaper.fit_transform(X_train_train)
        X_train_val = self.input_shaper.transform(X_train_val)
        
        start_time = time.time()
        for i, params_set in enumerate(self.parameters_list):
            if (models_ix is not None) and (i not in models_ix):
                continue
            eta = get_eta_repr(time.time() - start_time, i, len(self.parameters_list))
            print_bis(f"Split #{split_num+1}/{self.n_splits} - {step_name} Model #{i+1}/{len(self.parameters_list)} [ETA: {eta}]")
            model = self.blueprint(max_iter=max_iter, random_state=self.seed, **params_set)
            model.fit(X_train_train, y_train_train)
            self.models[i, split_num] = model
            if score_train:
                self.train_scores[i, split_num] = custom_score(model.predict(X_train_train), y_train_train)
            self.validation_scores[i, split_num] = custom_score(model.predict(X_train_val), y_train_val)
            if max_iter == -1:
                self.has_converged[i, split_num] = True
        
    def warm_up(self):
        for split_num in range(len(self.splits)):
            self.train_on_split(split_num, models_ix=None,
                                max_iter=self.iterations_to_warm, step_name='WARM UP',
                                score_train=False
                               )
    
    def select_n_best_models(self, n):
        average_validation_score = np.mean(self.validation_scores, axis=1)
        best_models_ix = np.argsort(average_validation_score)[-n:]
        return best_models_ix
    
    def train_n_best_models_until_convergence(self, n, split_num=0):        
        best_models_ix = self.select_n_best_models(n)
        self.train_on_split(split_num, models_ix=best_models_ix, max_iter=-1, step_name="TRAINING UNTIL CONVERGENCE", score_train=True)
        results = []
        for bm_ix in best_models_ix:
            results.append({"model": self.models[bm_ix, split_num], 
                            "train_score": self.train_scores[bm_ix, split_num],
                            "validation_score": self.validation_scores[bm_ix, split_num]
                           })
        return results

from sklearn.svm import SVC

def make_input_for_svm_extreme(h5_file):
    return make_input(h5_file, features=FEATURES, quantiles=TAIL_QUANTILES, dist_char=False, truncate_dist=False)

svm_extreme_params_grid_rbf_and_sigmoid = ParameterGrid(
    {"kernel": ["rbf", "sigmoid"],
     "C": [0.01, 0.1, 1, 10, 100],
     "gamma": ["auto", "scale"]
    }
)

svm_extreme_params_grid_polynomial = ParameterGrid(
    {"kernel": ["poly"],
     "C": [0.01, 0.1, 1, 10, 100],
     "degree": [1, 2, 3, 4] #if max_iter != -1
    })

svm_extreme_params_list = list(svm_extreme_params_grid_rbf_and_sigmoid) + list(svm_extreme_params_grid_polynomial)

svm_extreme = PoolModels(
    input_maker=InputMaker(make_input_for_svm_extreme),
    n_splits=2,
    train_size=3,
    input_shaper=InputShaper(StandardScaler()),
    blueprint=SVC,
    parameters_list=svm_extreme_params_list,
    iterations_to_warm=100,
    seed=3
)

svm_extreme.warm_up()
svm_extreme.train_n_best_models_until_convergence(3)