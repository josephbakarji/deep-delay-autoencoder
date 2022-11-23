import os

from aesindy.config import ROOTPATH

params = {}
params['data_path'] = os.path.join(ROOTPATH, 'testcases/results/')
if not os.path.isdir(params['data_path']):
    os.makedirs(params['data_path'])

params['case'] = 'rando'
params['model'] = 'predator_prey'
params['tend'] = 20
params['dt'] = 0.001
params['tau'] = None # skip 

params['system_coefficients'] = None 
params['normalization'] = None 
params['latent_dim'] = 2

params['noise'] = 0.0
params['interpolate'] = False
params['interp_dt'] = 0.01
params['interp_kind'] = 'cubic'
params['interp_coefs'] = [21, 3]

params['n_ics'] = 5

params['train_ratio'] = 0.8
params['input_dim'] = 128 # Try 256
params['poly_order'] = 2
params['include_sine'] = False
params['exact_features'] = False # Overrides poly_order

params['svd_dim'] = None
params['scale'] = False 

params['ode_net'] = False
params['ode_net_widths'] = [1.5, 3]

# sequential thresholding parameters
params['coefficient_threshold'] = 1e-6 ## set to none for turning off RFE
params['threshold_frequency'] = 100
params['coefficient_initialization'] = 'random_normal'
params['fixed_coefficient_mask'] = False
params['fix_coefs'] = False
params['trainable_auto'] = True 
params['sindy_pert'] = 0.0

params['verbose'] = 0

# loss function weighting
params['loss_weight_rec'] = 1.0
params['loss_weight_sindy_z'] = 0.0001 
params['loss_weight_sindy_x'] = 0.001
params['loss_weight_sindy_regularization'] = 1e-5
params['loss_weight_integral'] = 0.0 
params['loss_weight_x0'] = 0.0  
params['loss_weight_layer_l2'] = 0.0
params['loss_weight_layer_l1'] = 0.0 

params['activation'] = 'elu'
params['widths_ratios'] = [0.5, 0.25]
params['use_bias'] = True 

# training parameters
params['batch_size'] = 32 
params['learning_rate'] = 1e-3
params['sindy_learning_rate'] = 7e-3
params['learning_rate_sched'] = False 

params['save_checkpoints'] = False 
params['save_freq'] = 1

params['print_progress'] = True
params['print_frequency'] = 10 

# training time cutoffs
params['max_epochs'] = 3000 
params['patience'] = 100
params['sparse_weighting'] = None

params['sindycall_freq'] = 1
params['use_sindycall'] = False
params['sindy_threshold'] = 0.4
