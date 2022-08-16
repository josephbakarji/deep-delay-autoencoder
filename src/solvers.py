import numpy as np
from sklearn.model_selection import train_test_split
from scipy.integrate import odeint
from sindy_utils import library_size
from dynamical_models import get_model
from helper_functions import get_hankel
from tqdm import tqdm
import pdb

class SynthData:
    def __init__(self, 
        model='lorenz',
        args=None, 
        noise=0.0, 
        input_dim=128,
        normalization=None, 
        poly_order=None):

        self.model = model
        self.args = args
        self.noise = noise
        self.input_dim = input_dim
        self.normalization = None 
        self.poly_order = poly_order

    def solve_ivp(self, f, z0, time):
        """ Scipy ODE solver, returns z and dz/dt """
        z = odeint(f, z0, time)
        dz = np.array([f(z[i], time[i]) for i in range(len(time))])
        return z, dz
    
    def run_sim(self, n_ics, tend, dt, z0_stat=None):
        """ Runs solver over multiple initial conditions and builds Hankel matrix """

        f, Xi, model_dim, z0_mean_sug, z0_std_sug = get_model(self.model, self.args, self.poly_order, self.normalization)
        self.normalization = self.normalization if self.normalization is not None else np.ones((model_dim,))
        if z0_stat is None:
            z0_mean = z0_mean_sug
            z0_std = z0_std_sug
        else:
            z0_mean, z0_std = z0_stat

        time = np.arange(0, tend, dt)
        z0_mean = np.array(z0_mean) 
        z0_std =  np.array(z0_std) 
        z0 = z0_std*(np.random.rand(n_ics, model_dim)-.5) + z0_mean 

        delays = len(time) - self.input_dim
        z_full, dz_full, H, dH = [], [], [], []
        print("generating solutions..")
        for i in tqdm(range(n_ics)):
            z, dz = self.solve_ivp(f, z0[i, :], time)
            z *= self.normalization
            dz *= self.normalization

            # Build true solution (z) and hankel matrices
            z_full.append( z[:-self.input_dim, :] )
            dz_full.append( dz[:-self.input_dim, :] )
            x = z[:, 0] + self.noise * np.random.randn(len(time),) # Assumes first dim measurement
            dx = dz[:, 0] + self.noise * np.random.randn(len(time),) # Assumes first dim measurement
            H.append( get_hankel(x, self.input_dim, delays) )
            dH.append( get_hankel(dx, self.input_dim, delays) )
        
        self.z = np.concatenate(z_full, axis=0)
        self.dz = np.concatenate(dz_full, axis=0)
        self.x = np.concatenate(H, axis=1) 
        self.dx = np.concatenate(dH, axis=1) 
        self.t = time
        self.sindy_coefficients = Xi.astype(np.float32)
