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

def get_hankel_svd(H, reduced_dim):
    U, s, VT = np.linalg.svd(H, full_matrices=False)
    rec_v = np.matmul(VT[:reduced_dim, :].T, np.diag(s[:reduced_dim]))
    return U, s, VT, rec_v