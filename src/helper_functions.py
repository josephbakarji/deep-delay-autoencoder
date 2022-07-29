import itertools
import numpy as np
import random 

def get_hyperparameter_list(hyperparams):
    def dict_product(dicts):
        return [dict(zip(dicts, x)) for x in itertools.product(*dicts.values())]
    hyperparams_list = dict_product(hyperparams)
    random.shuffle(hyperparams_list)
    return hyperparams_list

def get_hankel(x, dimension, delays, skip_rows=1):
    if skip_rows>1:
        delays = len(x) - delays * skip_rows
    H = np.zeros((dimension, delays))
    for j in range(delays):
        H[:, j] = x[j*skip_rows:j*skip_rows+dimension]
    return H
