import pdb
import numpy as np
from aesindy.solvers import SynthData
from aesindy.training import TrainModel
from default_params import params

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


params['model'] = 'pendulum'
params['case'] = 'basic'
params['system_coefficients'] = [10, 10]
params['noise'] = 0.0
params['input_dim'] = 80
params['dt'] = np.sqrt(params['system_coefficients'][0]/params['system_coefficients'][1])/40
params['tend'] = 30 
params['n_ics'] = 35
params['poly_order'] = 1
params['include_sine'] = True
params['fix_coefs'] = False
params['coefficient_initialization'] = 'true'

params['sindy_learning_rate'] = 0.01 
params['learning_rate'] = 1e-3

params['activation'] = 'elu'
params['widths_ratios'] = [0.5, 0.25]
params['use_bias'] = True 

params['save_checkpoints'] = True
params['save_freq'] = 10

params['print_progress'] = True
params['print_frequency'] = 1
params['verbose'] = 0

# training time cutoffs
params['max_epochs'] = 3000
params['patience'] = 70 

# loss function weighting
params['loss_weight_rec'] = 0.5
params['loss_weight_sindy_z'] = 0.0001 
params['loss_weight_sindy_x'] = 0.001
params['loss_weight_sindy_regularization'] = 3e-5
params['loss_weight_integral'] = 0.3  
params['loss_weight_x0'] = 0.01 
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
