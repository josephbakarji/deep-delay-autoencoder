import numpy as np
from sklearn.model_selection import train_test_split
from scipy.integrate import odeint
from scipy import interpolate
from scipy.signal import savgol_filter
from .dynamical_models import get_model
from .helper_functions import get_hankel
from tqdm import tqdm
import pdb



class SynthData:
    def __init__(self, 
        model='lorenz',
        args=None, 
        noise=0.0, 
        input_dim=128,
        normalization=None):

        self.model = model
        self.args = args
        self.noise = noise
        self.input_dim = input_dim
        self.normalization = None 

    def solve_ivp(self, f, z0, time):
        """ Scipy ODE solver, returns z and dz/dt """
        z = odeint(f, z0, time)
        dz = np.array([f(z[i], time[i]) for i in range(len(time))])
        return z, dz
    
    def run_sim(self, n_ics, tend, dt, z0_stat=None):
        """ Runs solver over multiple initial conditions and builds Hankel matrix """

        f, Xi, model_dim, z0_mean_sug, z0_std_sug = get_model(self.model, self.args, self.normalization)
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
        
        
        
        

class RealData:
    def __init__(self, 
                input_dim=128,
                interpolate=False,
                interp_dt=0.01,
                savgol_interp_coefs=[21, 3],
                interp_kind='cubic'):

        self.input_dim = input_dim
        self.interpolate = interpolate 
        self.interp_dt = interp_dt 
        self.savgol_interp_coefs = savgol_interp_coefs
        self.interp_kind = interp_kind
    
    def build_solution(self, data):
        n_realizations = len(data['x'])
        dt = data['dt']
        if 'time' in data.keys():
            times = data['time']
        elif 'dt' in data.keys():
            times = []
            for xr in data['x']:
                times.append(np.linspace(0, dt*len(xr), len(xr), endpoint=False))
        
        x = data['x']
        if 'dx' in data.keys():
            dx = data['dx']
        else:
            dx = [np.gradient(xr, dt) for xr in x]
        
        new_times = []
        if self.interpolate:
            new_dt = self.interp_dt # Include with inputs
            print('old dt = ', dt)
            print('new dt = ', new_dt)
                    
            # Smoothing and interpolation
            for i in range(n_realizations):
                a, b = self.savgol_interp_coefs
                x[i] = savgol_filter(x[i], a, b)
                if 'dx' in data.keys():
                    dx[i] = savgol_filter(dx[i], a, b)

                t = np.arange(times[i][0], times[i][-2], new_dt)
                f = interpolate.interp1d(times[i], x[i], kind=self.interp_kind)
                x[i] = f(t) 
                df = interpolate.interp1d(times[i], dx[i], kind=self.interp_kind)
                dx[i] = df(t)
                    
                times[i] = t
#             new_times = np.array(new_times)
                    
        n = self.input_dim 
        n_delays = n
        xic = []
        dxic = []
        for j, xr in enumerate(x):
            n_steps = len(xr) - self.input_dim 
            xj = np.zeros((n_steps, n_delays))
            dxj = np.zeros((n_steps, n_delays))
            for k in range(n_steps):
                xj[k, :] = xr[k:n_delays+k]
                dxj[k, :] = dx[j][k:n_delays+k]
            xic.append(xj)
            dxic.append(dxj)
        H = np.vstack(xic)
        dH = np.vstack(dxic)
        
        self.t = np.hstack(times)
        self.x = H.T
        self.dx = dH.T
        self.z = np.hstack(x) 
        self.dz = np.hstack(dx)
        self.sindy_coefficients = None # unused
                
#         # Align times
#         for i in range(1, n_realizations):
#             if times[i] - times[i-1] >= dt*2:
#                 new_time[i] = new_time[i-1] + dt
