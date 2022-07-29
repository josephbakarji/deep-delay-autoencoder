import numpy as np
from scipy.integrate import odeint
from scipy.signal import savgol_filter
from scipy.special import legendre, chebyt
from scipy import interpolate
import sys
sys.path.append('../src')
from sindy_utils import library_size
from data_manage import DataStruct
import pdb
import json


class LorenzWW:
    # Can use inheritence 
    def __init__(self, 
            noise=0.0, # Not used in this case
            linear=False, # Not used in this case 
            input_dim=128,
            filename='./data/lorenzww.json',
            coefficients=[10, 8/3, 28.],
            normalization=[1/40, 1/40, 1/40], 
            interpolate=False,
            interp_dt=0.01,
            poly_order=3):

        self.option = 'delay'
        self.filename = filename
        self.input_dim = input_dim
#         self.coefficients = coefficients 
        self.sigma = coefficients[0]
        self.beta = coefficients[1]
        self.rho = coefficients[2]
        self.normalization = np.array(normalization) if normalization is not None else np.array([1, 1, 1])
        self.poly_order = poly_order
        self.interpolate = interpolate 
        self.interp_dt = interp_dt 

    
    def get_solution(self, tau=None):
        output_json = json.load(open(self.filename))
        times = np.array(output_json['times'])
        omegas = np.array(output_json['omegas'])
        domegas = np.array(output_json['domegas'])
        print(len(times))
        
        new_times = []
        if self.interpolate:
            new_dt = self.interp_dt # Include with inputs
            # Smoothing and interpolation
            for i in range(len(omegas)):
                omegas[i] = savgol_filter(omegas[i], 21, 3)
                domegas[i] = savgol_filter(domegas[i], 21, 3)

                times_new = np.arange(times[i][0], times[i][-2], new_dt)
                f = interpolate.interp1d(times[i], omegas[i], kind='cubic')
                omegas[i] = f(times_new)   # use interpolation function returned by `interp1d`
                df = interpolate.interp1d(times[i], domegas[i], kind='cubic')
                domegas[i] = df(times_new)   # use interpolation function returned by `interp1d`
                new_times.append(times_new)
            new_times = np.array(new_times)
        else:
            new_times = times
            new_dt = times[0][1] - times[0][0]
            
        
        dt = new_dt 
        n_ics = len(omegas) 
        d = 3
            
        n = self.input_dim 
        n_delays = n
        xic = []
        dxic = []
        for j, om in enumerate(omegas):
            n_steps = len(om) - self.input_dim # careful consistency 
            xj = np.zeros((n_steps, n_delays))
            dxj = np.zeros((n_steps, n_delays))
            for k in range(n_steps):
                xj[k, :] = om[k:n_delays+k]
                dxj[k, :] = domegas[j][k:n_delays+k]
            xic.append(xj)
            dxic.append(dxj)
        x = np.vstack(xic)
        dx = np.vstack(dxic)
        
        t = np.hstack(new_times)
        self.omega = np.hstack(omegas)
        self.domega = np.hstack(domegas)
                
        # Align times
        dt = t[1]-t[0]
        new_time = t.copy()
        for i in range(1, len(t)):
            if new_time[i] - new_time[i-1] >= dt*2:
                new_time[i] = new_time[i-1] + dt

                
        # Can be made a object rather than dictionary (part of class)
        data = DataStruct(name='measurements')
        data.t = new_time
        data.x = x 
        data.dx = dx
        data.ddx = None 
        data.z = omegas 
        data.dz = domegas
        data.ddz = None 
        data.sindy_coefficients = self.lorenz_coefficients()
        if self.option == 'projection':
            data.y_spatial = y_spatial
            data.modes = modes


        return data

    def lorenz_coefficients(self):
        """
        Generate the SINDy coefficient matrix for the Lorenz system.

        Arguments:
            normalization - 3-element list of array specifying scaling of each Lorenz variable
            poly_order - Polynomial order of the SINDy model.
            sigma, beta, rho - Parameters of the Lorenz system
        """
        Xi = np.zeros((library_size(3, self.poly_order), 3))
        Xi[1,0] = -self.sigma
        Xi[2,0] = self.sigma*self.normalization[0]/self.normalization[1]
        Xi[1,1] = self.rho*self.normalization[1]/self.normalization[0]
        Xi[2,1] = -1
        Xi[6,1] = -self.normalization[1]/(self.normalization[0]*self.normalization[2])
        Xi[3,2] = -self.beta
        Xi[5,2] = self.normalization[2]/(self.normalization[0]*self.normalization[1])
        return Xi

