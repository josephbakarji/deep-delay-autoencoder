# %%
import sys
sys.path.append("../../src")
sys.path.append("../")
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from lorenz import Lorenz
from training import TrainModel, get_callbacks, load_model
from basic_params import params
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
params['sindy_pert'] = 5.0 
params['max_epochs'] = 3000
params['loss_weight_x0'] = 0.2 
params['loss_weight_sindy_x'] = 0.001 
params['loss_weight_sindy_z'] = 0.001 
params['loss_weight_sindy_regularization'] = 1e-5
params['loss_weight_integral'] = 0.3
params['loss_weight_reconstruction'] = 0.1
params['svd_dim'] = None # try 7
params['scale'] = False  # try true
params['widths_ratios'] = [0.5, 0.25]
params['sparse_weighting'] = None 
params['normalization'] = [1/40, 1/40, 1/40] 
params['data_path'] = '/home/joebakarji/delay-auto/main/examples/data/'
params['loss_weight_layer_l2'] = 1e-7
params['loss_weight_layer_l1'] = 0.0 
params['use_bias'] = True 

## UNLock parameters
params['save_checkpoints'] = True
params['save_freq'] = 1
params['patience'] = 30 
params['fix_coefs'] = False
params['trainable_auto'] = True


hyperparams = {
'loss_weight_integral': [0.0, 1.0],
'sindy_pert' : [0.0, 1.0, 3.0, 5.0, 7.0, 10.0, 12.0, 13.5, 15.0, 20.0],
'widths_ratios' : [[0.5, 0.25]],
'learning_rate' : [2e-3],
'dt': [0.001],
'poly_order' : [2]
}

hyperparams_list = get_hyperparameter_list(hyperparams)
for hyperp in hyperparams_list:
    for key, val in hyperp.items():
        params[key] = val

    # Generate data
    data = generate_data(params)

    trainer = TrainModel(data, params)
    print(trainer.savename)


    train_data, test_data = trainer.get_data()
    trainer.save_params()
    print(trainer.params)

    # Create directory and file name
    os.makedirs(os.path.join(trainer.params['data_path'], trainer.savename), exist_ok=True)
    os.makedirs(os.path.join(trainer.params['data_path'], trainer.savename, 'checkpoints'), exist_ok=True)

    # LOAD h2v encoder and decoder
    filename = 'results_202111012031_lorenz_h2v_evolution_encoder'
    checkpoint_path = trainer.params['data_path']+filename+'/checkpoints/'
    cp_files = get_checkpoint_names(checkpoint_path)
    dec = []
    enc = []
    for name in cp_files:
        nametype = name.split('-')[1]
        if  nametype == 'dec':
            dec.append(name)
        elif nametype == 'enc':
            enc.append(name)
    dec.sort()
    enc.sort()
    dec_conv = dec[-1]
    enc_conv = enc[-1] 


    # Get model AFTER setting parameters
    trainer.model = trainer.get_model()
    trainer.model.predict(test_data)
    trainer.model.encoder.load_weights(checkpoint_path + enc_conv + '.ckpt').expect_partial()
    trainer.model.decoder.load_weights(checkpoint_path + dec_conv + '.ckpt').expect_partial()

    # Build model and fit
    optimizer = tf.keras.optimizers.Adam(lr=trainer.params['learning_rate'])
    trainer.model.compile(optimizer=optimizer, loss='mse')

    callback_list = get_callbacks(trainer.params, trainer.savename)
    trainer.history = trainer.model.fit(
            x=train_data, y=train_data, 
            batch_size=trainer.params['batch_size'],
            epochs=trainer.params['max_epochs'], 
            validation_data=(test_data, test_data),
            callbacks=callback_list,
            shuffle=True)

    # Save locked model
    prediction = trainer.model.predict(test_data)
    trainer.save_results(trainer.model)
