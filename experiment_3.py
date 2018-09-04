import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split, PredefinedSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.linear_model import RidgeClassifier
from sklearn.semi_supervised import LabelPropagation

from self_training import SelfTraining
from utils import generate_data, ToArray


ITERATIONS_NB = 5 #5, 10, 25, 50, 75, 100
POOL_SIZE = 10 #10, 50, 100

# Same parameters for each experiment
NB_PER_CAT = 25
TRAINING_WAY = 2 #0, 1, 2 
SEED = 42
TEST_SIZE = 0.2
CV = 4
n_iter = 40 #50 initially 
scoring = 'f1_micro'
n_jobs = -1
verbose = 3


# -- Loading dataset :
data = pd.read_csv("data/complete_enron.csv")
X = data.body
y = data.target

u_index, l_index = generate_data(X, y, NB_PER_CAT, SEED)
print("Data PER category:", np.unique(y[l_index], return_counts=True))

# -- Building dataset : labeled data and then unlabeled, need index reset
X_train, X_test, y_train, y_test = train_test_split(
    X.loc[l_index], y.loc[l_index], test_size=TEST_SIZE, stratify=y.loc[l_index])

X_train_u = pd.concat([X_train, X.loc[u_index]], axis=0)
y_train_u = pd.concat([y_train, pd.Series([-1] * len(u_index))], axis=0)

X_train_u.reset_index(drop=True, inplace=True)
y_train_u.reset_index(drop=True, inplace=True)

# Kfold validation with unlabeled data, mandatory for semi-supervised + gridsearch
test_size = int(len(l_index) / CV)
test_fold = np.array([-1] * y_train_u.shape[0]) # major issue corrected
for i in range(CV):
    test_fold[i*test_size:(i+1)*test_size] = i #i-th test with these samples

# Each test i is done with samples with label i
# Samples with -1 is used within every training
cross_validation = PredefinedSplit(test_fold)
classifier = SelfTraining()

X_train, y_train = X_train_u, y_train_u


# -- Building pipeline
pipeline = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', classifier)
])

params_grid = {
    'vect__stop_words': (None, "english"),
    'vect__strip_accents': ('ascii', None),
    'vect__analyzer': ('word', 'char'),
    'vect__ngram_range': ((1,1), (1,2), (1,3), (1,4), (1,5), (1,6)),
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__max_features': (10000, 15000, 25000, 50000),
    'vect__use_idf': (True, False),
}

# params_grid["clf__iterations_nb"] = (1, 5, 10)
params_grid["clf__iterations_nb"] = [ITERATIONS_NB]
# params_grid["clf__pool_size"] = (10, 50, 100)
params_grid["clf__pool_size"] = [POOL_SIZE]
# params_grid["clf__training_way"] = (0, 1, 2)
params_grid["clf__training_way"] = [TRAINING_WAY]

# -- Running Grid
rnd_search = RandomizedSearchCV(
    estimator=pipeline, 
    param_distributions=params_grid,
    n_iter=n_iter,
    scoring=scoring, 
    n_jobs=n_jobs,
    cv=cross_validation,
    verbose=verbose,
    random_state=SEED,
    refit=True
    )

rnd_search.fit(X_train, y_train)

# -- Results
print("\nN_per_cat :", NB_PER_CAT)
print("\nBest model :", rnd_search.best_params_)
print("\nBest score :", f1_score(y_test, rnd_search.predict(X_test), average='micro'))
