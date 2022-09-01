import sys
sys.path.append('../src/')

import pdb
import json
import numpy as np
from training import TrainModel
from solvers import RealData
from default_params import params
from paths import ROOTPATH

from scipy.io import loadmat
import mat73

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


params['model'] = 'fluttering'
params['case'] = 'Re1000'
params['system_coefficients'] = None
params['noise'] = 0.0
params['input_dim'] = 20
params['widths_ratios'] = [0.5, 0.25]
params['poly_order'] = 2
params['include_sine'] = False
params['fix_coefs'] = False

params['interpolate'] = True
params['interp_dt'] = 0.015
params['interp_kind'] = 'cubic'
params['interp_coefs'] = [21, 3]

params['save_checkpoints'] = False
params['save_freq'] = 5 

params['print_progress'] = True
params['print_frequency'] = 5 

# training time cutoffs
params['max_epochs'] = 3
params['patience'] = 10 

# loss function weighting
params['loss_weight_rec'] = 0.3
params['loss_weight_sindy_z'] = 0.001 
params['loss_weight_sindy_x'] = 0.001
params['loss_weight_sindy_regularization'] = 1e-5
params['loss_weight_integral'] = 0.1  
params['loss_weight_x0'] = 0.01 
params['loss_weight_layer_l2'] = 0.0
params['loss_weight_layer_l1'] = 0.0 

## Read data
annots = loadmat(ROOTPATH+'data/small_files/NACA0012_Re1000_AoA35_2D_forces.mat')
cd = annots['CD'].flatten()
data = {
'dt': annots['dt'][0][0],
'time': [np.linspace(0, annots['dt'][0][0]*len(cd), len(cd), endpoint=False)],
'x': [cd]
}

params['dt'] = data['dt']
params['tend'] = data['time'][-1][-1]
params['n_ics'] = len(data['x'])


R = RealData(input_dim=params['input_dim'],
            interpolate=params['interpolate'],
            interp_dt=params['interp_dt'],
            interp_kind=params['interp_kind'],
            savgol_interp_coefs=params['interp_coefs'])
    
R.build_solution(data)

trainer = TrainModel(R, params)
trainer.fit() 
