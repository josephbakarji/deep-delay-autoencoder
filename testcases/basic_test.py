import sys
sys.path.append('../src/')

import pdb
from solvers import SynthData
from training import TrainModel
from default_params import params

S = SynthData(model=params['model'], 
                args=params['system_coefficients'], 
                noise=params['noise'], 
                input_dim=params['input_dim'], 
                normalization=params['normalization'],
                poly_order=params['poly_order'])
S.run_sim(params['n_ics'], params['tend'], params['dt'])

trainer = TrainModel(S, params)
trainer.fit() 

