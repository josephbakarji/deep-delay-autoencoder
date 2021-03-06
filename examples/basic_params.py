import os
import sys
sys.path.append('../../')

params = {}
params['case'] = 'lock_sindy'
params['model'] = 'lorenz'
params['option'] = 'delay'
params['tend'] = 2000
params['dt'] = 0.001
params['tau'] = None

params['system_coefficients'] = [10, 8/3, 28.]
params['normalization'] = [1/40, 1/40, 1/40]
params['latent_dim'] = 3

params['noise'] = 0.0
params['n_ics'] = 1
params['train_ratio'] = 0.8

params['svd_dim'] = None
params['scale'] = False 
params['input_dim'] = 128 # Try 256
params['model_order'] = 1
params['poly_order'] = 2
params['include_sine'] = False
params['exact_features'] = False # Overrides poly_order
params['ode_net'] = False
params['ode_net_widths'] = [1.5, 3]
params['interpolate'] = False

# sequential thresholding parameters
params['coefficient_threshold'] = 1e-6 ## set to none for turning off RFE
params['threshold_frequency'] = 100
params['coefficient_initialization'] = 'true'
params['fixed_coefficient_mask'] = False
params['fix_coefs'] = True
params['trainable_auto'] = True 
params['sindy_pert'] = 0.0

# loss function weighting
## TODO: renormalize weights
params['loss_weight_rec'] = 1.0
params['loss_weight_sindy_z'] = 0.0001 
params['loss_weight_sindy_x'] = 0.001
params['loss_weight_sindy_regularization'] = 1e-5
params['loss_weight_integral'] = 0.0 #0.5 * 1/params['input_dim'] 
params['loss_weight_x0'] = 0.0 #0.2 # Forces first element to be the same in x and z
params['loss_weight_layer_l2'] = 0.0
params['loss_weight_layer_l1'] = 0.0 

params['activation'] = 'elu'
params['widths_ratios'] = [0.5, 0.25]
params['use_bias'] = True 

# training parameters
params['batch_size'] = 32 
params['learning_rate'] = 1e-3
params['learning_rate_sched'] = False 

params['save_checkpoints'] = False 
params['save_freq'] = 1

params['data_path'] = '/home/joebakarji/delay-auto/main/examples/data/'
params['print_progress'] = True
params['print_frequency'] = 10 

# training time cutoffs
params['max_epochs'] = 3000 
params['patience'] = 100
params['tensorboard'] = False
params['sparse_weighting'] = None

params['sindycall_freq'] = 1
params['use_sindycall'] = False
params['sindy_threshold'] = 0.4
