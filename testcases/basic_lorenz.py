# %%
import sys
sys.path.append("../../src")
sys.path.append("../")
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from training import TrainModel, get_callbacks, load_model
from default_params import params
from basic_run import generate_data, get_hyperparameter_list
from analyze import get_checkpoint_names

params['case'] = 'initv_intloss'
params['model'] = 'lorenz'
params['input_dim'] = 128 
params['tend'] = 30 
params['dt'] = 0.001
params['n_ics'] = 30 
params['batch_size'] = 500 
params['coefficient_initialization'] = 'true'
params['sindy_pert'] = 1.0 
params['max_epochs'] = 30
params['loss_weight_x0'] = 0.2 
params['loss_weight_sindy_x'] = 0.001 
params['loss_weight_sindy_z'] = 0.001 
params['loss_weight_sindy_regularization'] = 1e-5
params['loss_weight_integral'] = 0.5
params['loss_weight_reconstruction'] = 0.5
params['svd_dim'] = None # try 7
params['scale'] = False  # try true
params['widths_ratios'] = [0.5, 0.25]
params['sparse_weighting'] = None 
params['normalization'] = [1/40, 1/40, 1/40] 
params['loss_weight_layer_l2'] = 1e-7
params['loss_weight_layer_l1'] = 0.0 
params['use_bias'] = True 

## UNLock parameters
params['save_checkpoints'] = True
params['save_freq'] = 5
params['patience'] = 30 
params['fix_coefs'] = False
params['trainable_auto'] = True
