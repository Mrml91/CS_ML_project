from sklearn.decomposition import PCA
import pandas as pd

class CustomPCA(PCA):
    
    sep = "ù`£ls:"
    
    def __init__(self, columns_filter, name="custom_PCA", var_capture=0.95, **kwargs):
        self.columns_filter = columns_filter
        self.name = name
        self.pca = PCA(var_capture, copy=True, **kwargs)
        self.copy = self.pca.copy # needed to pass a check
        
    def get_sub(self, X):
        matching_cols = [c for c in X.columns if self.columns_filter(c)]
        return X.loc[:, matching_cols]
    
    def get_compl_sub(self, X):
        col_selector = [c for c in X.columns if not self.columns_filter(c)]
        return X.loc[:, col_selector]
    
    def generate_col_names(self, n):
        return [(self.name, str(i)) for i in range(n)]
    
    def col_tuples_to_str(self, cols):
        return [self.sep.join(c) for c in cols]
    
    def col_str_to_tuples(self, cols):
        return pd.MultiIndex.from_tuples([c.split(self.sep) for c in cols])
        
    
    def fit(self, X):
        subX = self.get_sub(X)
        self.pca.fit(subX)
        
    def _transform(self, X, fit=False):
        subX, otherX = self.get_sub(X), self.get_compl_sub(X)
        if fit:
            subX_transformed = self.pca.fit_transform(subX)
        else:
            subX_transformed = self.pca.transform(subX)
        new_col_names = self.generate_col_names(subX_transformed.shape[1])
        # trick because assign doesn't support tuple column names
        otherX.columns = self.col_tuples_to_str(otherX.columns)
        new_col_names = self.col_tuples_to_str(new_col_names)
        newX = otherX.assign(**{new_col_names[i]: subX_transformed[:, i] for i in range(subX_transformed.shape[1])})
        newX.columns = self.col_str_to_tuples(newX.columns)
        return newX
    
    def fit_transform(self, X):
        return self._transform(X, fit=True)
    
    def transform(self, X):
        return self._transform(X, fit=False)
        
        
        