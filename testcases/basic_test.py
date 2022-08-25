import sys
sys.path.append('../src/')

import pdb
import numpy as np
from solvers import SynthData
from training import TrainModel
from default_params import params

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


params['model'] = 'pendulum'
params['case'] = 'basic'
params['system_coefficients'] = [9.8, 10]
params['noise'] = 0.0
params['input_dim'] = 80
params['dt'] = np.sqrt(params['system_coefficients'][0]/params['system_coefficients'][1])/params['input_dim']
params['tend'] = 2
params['n_ics'] = 30
params['poly_order'] = 1
params['include_sine'] = True
params['fix_coefs'] = False

params['save_checkpoints'] = True 
params['save_freq'] = 1 

params['print_progress'] = True
params['print_frequency'] = 5 

# training time cutoffs
params['max_epochs'] = 5
params['patience'] = 70 

# loss function weighting
params['loss_weight_rec'] = 1.0
params['loss_weight_sindy_z'] = 0.001 
params['loss_weight_sindy_x'] = 0.001
params['loss_weight_sindy_regularization'] = 1e-5
params['loss_weight_integral'] = 0.1  
params['loss_weight_x0'] = 0.1 
params['loss_weight_layer_l2'] = 0.0
params['loss_weight_layer_l1'] = 0.0 


S = SynthData(model=params['model'], 
                args=params['system_coefficients'], 
                noise=params['noise'], 
                input_dim=params['input_dim'], 
                normalization=params['normalization'])
S.run_sim(params['n_ics'], params['tend'], params['dt'])

trainer = TrainModel(S, params)
trainer.fit() 

