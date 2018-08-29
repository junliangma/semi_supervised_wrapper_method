import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_X_y
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix


class SelfTraining(BaseEstimator, TransformerMixin):
    def __init__(self, learner=LogisticRegression(), iterations_nb=5, pool_size=100, training_way=0, random_state=None):
        self.learner = learner
        self.iterations_nb = iterations_nb
        self.pool_size = pool_size
        self.training_way = training_way #0: Hard, 1: Soft, 2: Soft multi
        self.random_state = random_state
        self._labeled_data_size = -1
        self._unlabeled_data_size = -1
        self._categories_nb = -1

    def get_params(self, deep=False): # deep=False due to a little issue in Scikit-Learn
        return {"learner": self.learner,
            "iterations_nb": self.iterations_nb,
            "pool_size": self.pool_size,
            "training_way": self.training_way,
            "random_state": self.random_state}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        
        return self

    def __x_y_preparation(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'])
        
        self._categories_nb = len(np.unique(y))

        l_index = np.argwhere(y != -1)
        u_index = np.argwhere(y == -1)
        
        self._labeled_data_size = len(np.argwhere(y != -1))
        self._unlabeled_data_size = len(np.argwhere(y == -1))

        X_new_shape = ((self._labeled_data_size + self._categories_nb * self._unlabeled_data_size), X.shape[1])
        y_new_shape = ((self._labeled_data_size + self._categories_nb * self._unlabeled_data_size),)
        X_new = lil_matrix(X_new_shape, dtype=np.float64)
        y_new = np.zeros(y_new_shape)

        i_l = 0
        i_u = 0
        for i, y_row in enumerate(y):
            X_row = X[i]
            y_row = y_row
            
            # label
            if i in l_index:
                X_new[i_l] = X_row
                y_new[i_l] = y_row
                i_l += 1
            
            # unlabel
            elif i in u_index:
                # we need all possible category for each unlabeled data
                for cat in range(self._categories_nb):
                    local_i = (self._labeled_data_size) + (cat * self._unlabeled_data_size) + i_u
                    X_new[local_i] = X_row
                    y_new[local_i] = cat
                i_u += 1
                
        return X_new, y_new

    def __m_step(self, X, y, weights):
        self.learner.fit(X, y, sample_weight=weights)

        return self

    def __e_step(self, X, weights):

        if not getattr(self.learner, "decision_function", None):
            raise ValueError('No decision_function in model.')

        for i in np.random.randint(self._labeled_data_size,
                                   self._labeled_data_size + self._unlabeled_data_size,
                                   size=self.pool_size):

            scores = self.learner.decision_function(X[i])
            predicted_cat = scores.argmax(axis=1)[0]
            
            # Computing weights
            new_weights = np.zeros(self._categories_nb)
            if self.training_way == 0: # Hard
                new_weights[predicted_cat] = 1

            elif self.training_way == 1: # Soft
                probs = np.exp(scores) / np.sum(np.exp(scores))
                new_weights[predicted_cat] = probs[0][predicted_cat] # only the best weight, not any others
            
            elif self.training_way == 2: # Soft multi
                probs = np.exp(scores) / np.sum(np.exp(scores))
                new_weights = probs[0] # all weights
            
            # Updating weights in each category for X[i]
            for cat in range(self._categories_nb):
                weights[i + cat * self._unlabeled_data_size] = new_weights[cat]

        return weights

    def fit(self, X, y):
        np.random.seed(self.random_state)

        X, y = self.__x_y_preparation(X, y)

        weights = np.array(self._labeled_data_size * [1] + self._categories_nb * self._unlabeled_data_size * [0], dtype=np.float64)

        self.__m_step(X, y, weights)
        for _ in range(self.iterations_nb):
            weights = self.__e_step(X, weights)
            self.__m_step(X, y, weights)

        return self

    def predict(self, X):
        return self.learner.predict(X)

    def decision_function(self, X):
        return self.learner.decision_function(X)

    def score(self, X, y):
        return self.learner.score(X, y)


if __name__ == '__main__':
    pass
