# %%
import sys
sys.path.append("../../src")
sys.path.append("../")
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from __init__ import *
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from lorenz import Lorenz
from training import TrainModel, get_callbacks
from basic_params import params
from basic_run import generate_data, train, get_hyperparameter_list


params['case'] = 'h2v_evolution'
params['model'] = 'lorenz'
params['input_dim'] = 128 
params['tend'] = 60
params['dt'] = 0.001
params['n_ics'] = 50
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
params['widths_ratios'] = [0.5, 0.25]
params['sparse_weighting'] = None 
params['normalization'] = [1/40, 1/40, 1/40] 
params['data_path'] = '/home/joebakarji/delay-auto/main/examples/data/'
params['loss_weight_layer_l2'] = 0.0 
params['loss_weight_layer_l1'] = 0.0 
params['use_bias'] = True 
params['learning_rate'] = 5e-4

# Generate data
data = generate_data(params)
trainer = TrainModel(data, params)

## Lock parameters
encdec_patience = 10
trainer.params['case'] = 'h2v_evolution'
trainer.params['save_checkpoints'] = True
trainer.params['patience'] = 40 
trainer.params['fix_coefs'] = True
trainer.params['trainable_auto'] = True

trainer.savename = trainer.get_name()
print(trainer.savename)

train_data, test_data = trainer.get_data()
trainer.save_params()
print(trainer.params)

# Get model AFTER setting parameters
trainer.model = trainer.get_model()

## Get SVD output
reduced_dim = 3
U, s, VT = np.linalg.svd(data.x.T, full_matrices=False)
v = np.matmul(VT[:reduced_dim, :].T, np.diag(s[:reduced_dim]))

# Create directory and file name
os.makedirs(os.path.join(trainer.params['data_path'], trainer.savename), exist_ok=True)
os.makedirs(os.path.join(trainer.params['data_path'], trainer.savename, 'checkpoints'), exist_ok=True)

########################

# ENCODER Checkpoints
checkpoint_path_encoder = os.path.join(trainer.params['data_path'], trainer.savename, 'checkpoints', 'cp-enc-{epoch:04d}.ckpt')
cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path_encoder, 
                verbose=1, 
                save_weights_only=True,
                save_freq=params['save_freq'] * int(trainer.params['tend']/trainer.params['dt']*trainer.params['n_ics']/ \
                                    trainer.params['batch_size'] * trainer.params['train_ratio']))


# ENCODER TRAINING
optimizer = tf.keras.optimizers.Adam(lr=trainer.params['learning_rate'])
trainer.model.encoder.compile(optimizer=optimizer, loss='mse')

history_encoder = trainer.model.encoder.fit(
        x=data.x, y=v, 
        batch_size=trainer.params['batch_size'],
        epochs=20,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=encdec_patience, monitor='loss'), cp_callback],
        shuffle=True)


########################

# DECODER Checkpoints
checkpoint_path_decoder = os.path.join(trainer.params['data_path'], trainer.savename, 'checkpoints', 'cp-dec-{epoch:04d}.ckpt')
cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path_decoder, 
                verbose=1, 
                save_weights_only=True,
                save_freq=trainer.params['save_freq'] * int(trainer.params['tend']/trainer.params['dt']*trainer.params['n_ics']/ \
                                    trainer.params['batch_size'] * trainer.params['train_ratio']))

# DECODER TRAINING
trainer.model.decoder.compile(optimizer=optimizer, loss='mse')
history_decoder = trainer.model.decoder.fit(
        x=v, y=data.x, 
        batch_size=trainer.params['batch_size'],
        epochs=20,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=encdec_patience, monitor='loss'), cp_callback],
        shuffle=True)

########################

# FULL MODEL Checkpoints
checkpoint_path = os.path.join(trainer.params['data_path'], trainer.savename, 'checkpoints', 'cp-{epoch:04d}.ckpt')
cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path, 
                verbose=1, 
                save_weights_only=True,
                save_freq=trainer.params['save_freq'] * int(trainer.params['tend']/trainer.params['dt']*trainer.params['n_ics']/ \
                                    trainer.params['batch_size'] * trainer.params['train_ratio']))


# Build model and fit
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

