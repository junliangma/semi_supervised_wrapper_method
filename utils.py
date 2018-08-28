import random
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

def generate_data(X, y, nb_per_cat, seed):
    # Shuffling list
    index = list(range(y.shape[0]))
    random.seed(seed)
    random.shuffle(index)
    u_index, l_index = [], []
    
    distribution_dict = {i:0 for i in np.unique(y)}
    
    for i in index:
        cat = y[i]
        if distribution_dict[cat] < nb_per_cat:
            l_index.append(i)
            distribution_dict[cat] = distribution_dict[cat] + 1
        
        else:
            u_index.append(i)
            
    for key, value in distribution_dict.items():
        if value < nb_per_cat:
            print("Warning: not enough data for category:", key)

    print("Labeled data PER category:", np.unique(y[l_index], return_counts=True))

    return np.array(u_index), np.array(l_index)

class ToArray(BaseEstimator, TransformerMixin):
    """ ToArray """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray()