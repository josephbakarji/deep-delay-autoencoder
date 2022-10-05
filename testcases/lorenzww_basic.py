import pdb
import json
import numpy as np
from aesindy.training import TrainModel
from aesindy.solvers import RealData
from default_params import params
from aesindy.config import ROOTPATH

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


params['model'] = 'lorenzww'
params['case'] = 'basic'
params['system_coefficients'] = None
params['noise'] = 0.0
params['input_dim'] = 80
params['poly_order'] = 2
params['include_sine'] = False
params['fix_coefs'] = False

params['interpolate'] = True
params['interp_dt'] = 0.01
params['interp_kind'] = 'cubic'
params['interp_coefs'] = [21, 3]

params['save_checkpoints'] = True 
params['save_freq'] = 5 

params['print_progress'] = True
params['print_frequency'] = 5 

# training time cutoffs
params['max_epochs'] = 300
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
output_json = json.load(open(os.path.join(ROOTPATH, 'data/lorenzww.json')))
data = {
'time': [np.array(time) for time in output_json['times']],
'dt': output_json['times'][0][1]-output_json['times'][0][0],
'x': [np.array(x) for x in output_json['omegas']],
'dx': [np.array(x) for x in output_json['domegas']],
}

params['dt'] = data['dt']
params['tend'] = data['time'][-1][-1] 
params['n_ics'] = len(data['x'])


R = RealData(input_dim=params['input_dim'],
             interpolate=params['interpolate'],
             interp_dt=params['interp_dt'],
             savgol_interp_coefs=params['interp_coefs'],
             interp_kind=params['interp_kind'])
    
R.build_solution(data)

pdb.set_trace()
trainer = TrainModel(R, params)
trainer.fit() 
