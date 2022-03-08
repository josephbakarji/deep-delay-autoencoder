# %%
import sys
sys.path.append("../src")
import os
import pdb

import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from lorenz import Lorenz
from waterlorenz import LorenzWW
from predprey import PredPrey
from rossler import Rossler 

import itertools
import random 
from basic_params import params

def get_hyperparameter_list(hyperparams):
    def dict_product(dicts):
        return [dict(zip(dicts, x)) for x in itertools.product(*dicts.values())]
    hyperparams_list = dict_product(hyperparams)
    random.shuffle(hyperparams_list)
    return hyperparams_list

def generate_data(params):
    if params['model'] == 'lorenz':
        model = Lorenz
    elif params['model'] == 'predprey':
        model = PredPrey
    elif params['model'] == 'rossler':
        model = Rossler
    elif params['model'] == 'lorenzww':
        model = LorenzWW 
        L = model(option=params['option'], coefficients=params['system_coefficients'], noise=params['noise'], 
            input_dim=params['input_dim'], poly_order=params['poly_order'], filename='../data/lorenzww.json',
            interpolate=params['interpolate'])
        data = L.get_solution()
        return data
       
    L = model(option=params['option'], coefficients=params['system_coefficients'], noise=params['noise'], 
    input_dim=params['input_dim'], poly_order=params['poly_order'], normalization=params['normalization'])
    data = L.get_solution(params['n_ics'], params['tend'], params['dt'], tau=params['tau'])
    return data

if __name__ == "__main__":
    data = generate_data(params)
    trainer = TrainModel(data, params)
    trainer.fit() 
    
    