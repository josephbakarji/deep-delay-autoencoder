# %%
import sys
sys.path.append("../")
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from __init__ import *
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from aesindy.training import TrainModel, get_callbacks
from basic_params import params
from basic_run import generate_data, train, get_hyperparameter_list
from config import ROOTPATH


params['case'] = 'lockauto_unlockf'
params['model'] = 'lorenz'
params['input_dim'] = 128 
params['tend'] = 6
params['dt'] = 0.001
params['n_ics'] = 5
params['batch_size'] = 500 
params['coefficient_initialization'] = 'true' 
params['max_epochs'] = 3000
params['loss_weight_x0'] = 0.2 
params['loss_weight_sindy_x'] = 0.001 
params['loss_weight_sindy_z'] = 0.0001 #0.2 # forces first element to be the same in x and z
params['loss_weight_sindy_regularization'] = 1e-5
params['loss_weight_integral'] = 0.0
params['svd_dim'] = None # try 7
params['scale'] = False  # try true
params['widths_ratios'] = [0.75, 0.4, 0.2]
params['sparse_weighting'] = None 
params['normalization'] = [1/40, 1/40, 1/40] 
params['data_path'] = ROOTPATH
params['loss_weight_layer_l2'] = 0.0 
params['loss_weight_layer_l1'] = 0.0 
params['use_bias'] = True 


# Generate data
data = generate_data(params)

trainer = TrainModel(data, params)

## Lock parameters
trainer.params['case'] = 'lock_opt' + 'test2'
trainer.params['save_checkpoints'] = False
trainer.params['patience'] = 1
trainer.params['fix_coefs'] = True
trainer.params['trainable_auto'] = True

trainer.savename = trainer.get_name(include_date=False)
print(trainer.savename)

train_data, test_data = trainer.get_data()
trainer.save_params()
print(trainer.params)

# Create directory and file name
os.makedirs(os.path.join(trainer.params['data_path'], trainer.savename), exist_ok=True)
os.makedirs(os.path.join(trainer.params['data_path'], trainer.savename, 'checkpoints'), exist_ok=True)

# Get model AFTER setting parameters
trainer.model = trainer.get_model()

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


####### 
####### 

## UNLock parameters
trainer.params['case'] = 'lockauto_unlockf'
trainer.params['save_checkpoints'] = True
trainer.params['save_freq'] = 2
trainer.params['patience'] = 1 
trainer.params['fix_coefs'] = False
trainer.params['trainable_auto'] = False

trainer.savename = trainer.get_name(include_date=True)
trainer.save_params()

# Create directory and file name
os.makedirs(os.path.join(trainer.params['data_path'], trainer.savename), exist_ok=True)
os.makedirs(os.path.join(trainer.params['data_path'], trainer.savename, 'checkpoints'), exist_ok=True)

trainer.model_unlock = trainer.get_model()
trainer.model_unlock.predict(test_data) # For building model, required for transfer 
trainer.model_unlock.set_weights(trainer.model.get_weights()) # Transfer weights 
trainer.model_unlock.compile(optimizer=optimizer, loss='mse')

callback_list = get_callbacks(trainer.params, trainer.savename)
trainer.history = trainer.model_unlock.fit(
        x=train_data, y=train_data, 
        batch_size=trainer.params['batch_size'],
        epochs=trainer.params['max_epochs'], 
        validation_data=(test_data, test_data),
        callbacks=callback_list,
        shuffle=True)

prediction = trainer.model_unlock.predict(test_data)
trainer.save_results(trainer.model_unlock)

