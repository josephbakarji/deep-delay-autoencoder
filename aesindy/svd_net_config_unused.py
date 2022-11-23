import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
import pysindy as ps

class PreSVD_Sindy_Autoencoder(tf.keras.Model):
    def __init__(self, params, **kwargs):
        super(PreSVD_Sindy_Autoencoder, self).__init__(**kwargs)
        self.params = params
        self.latent_dim = params['latent_dim']
        self.input_dim = params['input_dim']
        self.svd_dim = params['svd_dim']
        self.widths = params['widths']
        self.activation = params['activation']
        self.library_dim = params['library_dim']
        self.poly_order = params['poly_order']
        self.include_sine = params['include_sine']
        self.initializer = params['coefficient_initialization'] # fix That's sindy's!
        self.epochs = params['max_epochs'] # fix That's sindy's!
        self.rfe_threshold = params['coefficient_threshold']
        self.rfe_frequency = params['threshold_frequency']
        self.print_frequency = params['print_frequency']
        self.sindy_pert = params['sindy_pert']
        self.fixed_coefficient_mask = None
        self.actual_coefs = params['actual_coefficients']
        self.use_bias = params['use_bias']
        self.l2 = params['loss_weight_layer_l2']
        self.l1 = params['loss_weight_layer_l1']
        if params['fixed_coefficient_mask']:
            self.fixed_coefficient_mask = tf.constant(value=np.abs(self.actual_coefs)>1e-10, dtype=tf.float32) 

        self.time = tf.constant(value=np.linspace(0.0, params['dt']*params['svd_dim'], params['svd_dim'], endpoint=False))
        self.dt = tf.constant(value=params['dt'], dtype=tf.float32)
        
        self.encoder = self.make_network(self.svd_dim, self.latent_dim, self.widths, name='encoder')
        self.decoder = self.make_network(self.latent_dim, self.svd_dim, self.widths[::-1], name='decoder')
        self.sindy = Sindy(self.library_dim, self.latent_dim, self.poly_order, model=params['model'], initializer=self.initializer,
                           actual_coefs=self.actual_coefs, rfe_threshold=self.rfe_threshold, include_sine=self.include_sine,
                           exact_features=params['exact_features'], fix_coefs=params['fix_coefs'], sindy_pert=self.sindy_pert)
    
    def make_network(self, input_dim, output_dim, widths, name):
        out_activation = 'linear'
        initializer= tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg", distribution="uniform")
        model = tf.keras.Sequential()
        for i, w in enumerate(widths):
            if i ==0:
                use_bias = self.use_bias
            else:
                use_bias = True 
            model.add(tf.keras.layers.Dense(w, activation=self.activation, kernel_initializer=initializer, name=name+'_'+str(i), use_bias=use_bias, kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2)))
        model.add(tf.keras.layers.Dense(output_dim, activation=out_activation, kernel_initializer=initializer, name=name+'_'+'out', use_bias=self.use_bias))
        return model
                
    def call(self, datain):
        x = datain[1]
        return self.decoder(self.encoder(x))

    def compile(self,
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss=tf.keras.losses.BinaryCrossentropy(),
                sindy_optimizer=None,
                **kwargs):
        super(AutoencoderSindy, self).compile(optimizer=optimizer, loss=loss, **kwargs)
        if sindy_optimizer is None:
            self.sindy_optimizer = tf.keras.optimizers.get(optimizer)
        else:
            self.sindy_optimizer = tf.keras.optimizers.get(sindy_optimizer)
    
    @tf.function
    def train_step(self, data):
        inputs, outputs = data
        x = inputs[0]
        v = inputs[1]
        dv_dt = tf.expand_dims(inputs[2], 2)
        x_out = outputs[0]
        v_out = outputs[1]
        dv_dt_out = tf.expand_dims(outputs[2], 2)

        # TRy data in tape and outside tape 
        with tf.GradientTape() as tape:
            loss, losses = self.get_loss(x, v, dv_dt, x_out, v_out, dv_dt_out)

        # split trainable variables for autoencoder and dynamics so that you can use seperate optimizers
        trainable_vars = self.encoder.trainable_weights + self.decoder.trainable_weights + \
                            self.sindy.trainable_weights
        n_sindy_weights = len(self.sindy.trainable_weights)
        grads = tape.gradient(total_loss, trainable_vars)
        grads_autoencoder = grads[:-n_sindy_weights]
        grads_sindy = grads[-n_sindy_weights:]

        self.optimizer.apply_gradients(zip(grads_autoencoder, trainable_weights[:-n_sindy_weights]))
        self.sindy_optimizer.apply_gradients(zip(grads_sindy, trainable_weights[-n_sindy_weights:]))

        ## Keep track and update losses
        self.update_losses(loss, losses)
        return {m.name: m.result() for m in self.metrics}

    
    @tf.function
    def test_step(self, data):
        inputs, outputs = data
        x = inputs[0]
        v = inputs[1]
        dv_dt = tf.expand_dims(inputs[2], 2)
        x_out = outputs[0]
        v_out = outputs[1]
        dv_dt_out = tf.expand_dims(outputs[2], 2)

        loss, losses = self.get_loss(x, v, dv_dt, x_out, v_out, dv_dt_out)
        
        ## Keep track and update losses
        self.update_losses(loss, losses)
        return {m.name: m.result() for m in self.metrics}
    

    @tf.function
    def get_loss(self, x, v, dv_dt, x_out, v_out, dv_dt_out):
        losses = {}
        loss = 0
        if self.params['loss_weight_sindy_z'] > 0.0:
            with tf.GradientTape() as t1:
                t1.watch(v)
                z = self.encoder(v, training=True)
            dz_dv = t1.batch_jacobian(z, v)
            dz_dt = tf.matmul( dz_dv, dv_dt )
        else:
            z = self.encoder(v, training=True)

        if self.params['loss_weight_sindy_x'] > 0.0:
            with tf.GradientTape() as t2:
                t2.watch(z)
                vh = self.decoder(z, training=True)
            dvh_dz = t2.batch_jacobian(vh, z)
            dz_dt_sindy = tf.expand_dims(self.sindy(z), 2)
            dvh_dt = tf.matmul( dvh_dz, dz_dt_sindy )
        else:
            vh = self.decoder(z, training=True) 

        # SINDy consistency loss
        if self.params['loss_weight_integral'] > 0.0:
            sol = z 
            loss_int = tf.square(sol[:, 0] - x[:, 0]) 
            total_steps = len(self.time)
            for i in range(1, len(self.time)):
                k1 = self.sindy(sol)
                k2 = self.sindy(sol + self.dt/2 * k1)
                k3 = self.sindy(sol + self.dt/2 * k2)
                k4 = self.sindy(sol + self.dt * k3)
                sol = sol + 1/6 * self.dt * (k1 + 2*k2 + 2*k3 + k4) 
                # To avoid nans - explicitely with tf.where(tf.is_nan()) not compatible with gradients 
                sol = tf.where(tf.abs(sol) > 500.00, 50.0, sol) 
                loss_int += tf.square(sol[:, 0] - x[:, i])
            loss += self.params['loss_weight_integral'] * loss_int / total_steps 
            losses['integral'] = loss_int


        if self.params['loss_weight_sindy_x'] > 0.0:
            loss_dx = tf.reduce_mean( tf.square(dvh_dt - dv_dt_out) ) 
            loss += self.params['loss_weight_sindy_x'] * loss_dx
            losses['sindy_x'] = loss_dx

        if self.params['loss_weight_sindy_z'] > 0.0:
            loss_dz = tf.reduce_mean( tf.square(dz_dt - dz_dt_sindy) )
            loss += self.params['loss_weight_sindy_z'] * loss_dz
            losses['sindy_z'] = loss_dz

        if self.params['loss_weight_x0'] > 0.0:
            loss_x0 = tf.reduce_mean( tf.square(z[:, 0] - x[:, 0]) )
            loss += self.params['loss_weight_x0'] * loss_x0
            losses['x0'] = loss_x0

        loss_rec = tf.reduce_mean( tf.square(vh - v_out) ) 
        loss_l1 = tf.reduce_mean( tf.abs(self.sindy.coefficients) )

        loss += self.params['loss_weight_rec'] * loss_rec \
              + self.params['loss_weight_sindy_regularization'] * loss_l1

        if self.fixed_coefficient_mask is not None:
            self.sindy.coefficients.assign( tf.multiply(self.sindy.coefficients, self.fixed_coefficient_mask) )

        losses['rec'] = loss_rec
        losses['l1'] = loss_l1
        
        return loss, losses
    
    @tf.function
    def update_losses(self, loss, losses):
        total_met.update_state(loss)
        rec_met.update_state(losses['rec'])
        if self.params['loss_weight_sindy_z'] > 0:
            sindy_z_met.update_state(losses['sindy_z'])
        if self.params['loss_weight_sindy_x'] > 0:
            sindy_x_met.update_state(losses['sindy_x'])
        if self.params['loss_weight_integral'] > 0:
            integral_met.update_state(losses['integral'])
        if self.params['loss_weight_x0'] > 0:
            x0_met.update_state(losses['x0'])
        if self.params['loss_weight_sindy_regularization'] > 0:
            l1_met.update_state(losses['l1'])

    # Check if needed
    @property
    def metrics(self):
        m = [total_met, rec_met]
        if self.params['loss_weight_sindy_z'] > 0.0:
            m.append(sindy_z_met)
        if self.params['loss_weight_sindy_x'] > 0.0:
            m.append(sindy_x_met)
        if self.params['loss_weight_integral'] > 0.0:
            m.append(integral_met)
        if self.params['loss_weight_x0'] > 0.0:
            m.append(x0_met)
        if self.params['loss_weight_sindy_regularization'] > 0.0:
            m.append(l1_met)
        return m 

