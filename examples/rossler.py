import numpy as np
from scipy.integrate import odeint
from scipy.special import legendre, chebyt
import sys
sys.path.append('../src')
from sindy_utils import library_size
from data_manage import DataStruct
import pdb

class Rossler:
    def __init__(self, 
            option='delay', 
            coefficients=[0.2, 0.2, 5.7], 
            noise=0.0, 
            input_dim=128,
            normalization=[1.0, 1.0, 1.0], # Not used in predator prey
            linear=False,
            poly_order=2,
            include_sine=False):

        self.option = option
        self.a = coefficients[0]
        self.b = coefficients[1]
        self.c = coefficients[2]
        self.noise = noise
        self.input_dim = input_dim
        self.normalization = np.array(normalization) if normalization is not None else np.array([1, 1])
        self.linear = linear
        self.poly_order = poly_order
        self.include_sine = include_sine

    
    def get_solution(self, n_ics, tend, dt, ic_means=[2, 2, 2], ic_widths=[-9, 0, 0], tau=None):
        """
        Generate a set of Lorenz training data for multiple random initial conditions.

        Arguments:
            n_ics - Integer specifying the number of initial conditions to use.
            noise_strength - Amount of noise to add to the data.

        Return:
            data - Dictionary containing elements of the dataset. See generate_lorenz_data()
            doc string for list of contents.
        """
        t = np.arange(0, tend, dt)
        n_steps = len(t) - self.input_dim
        if tau is not None:
            n_steps = len(t) - self.input_dim * int(tau/dt)
        
        ic_means = np.array(ic_means) 
        ic_widths = 2 * np.array(ic_widths) 

        # training data
        ics = ic_widths*(np.random.rand(n_ics, 3)-.5) + ic_means
        data = self.generate_data(ics, t, tau=tau)
            
        data.x = data.x.reshape((n_steps*n_ics, self.input_dim)) \
                + self.noise * np.random.randn(n_steps * n_ics, self.input_dim)  
        data.dx = data.dx.reshape((n_steps*n_ics, self.input_dim)) \
                + self.noise * np.random.randn(n_steps * n_ics, self.input_dim)
        data.ddx = data.ddx.reshape((n_steps*n_ics, self.input_dim)) \
                + self.noise * np.random.randn(n_steps * n_ics, self.input_dim)
        
        
        full_z = data.z[0, :-self.input_dim, :]
        full_dz = data.dz[0, :-self.input_dim, :]
        full_ddz = data.ddz[0, :-self.input_dim, :]
        for i in range(1, data.z.shape[0]):
            full_z = np.concatenate((full_z, data.z[i, :-self.input_dim, :]), axis=0)
            full_dz = np.concatenate((full_dz, data.dz[i, :-self.input_dim, :]), axis=0)
            full_ddz = np.concatenate((full_ddz, data.ddz[i, :-self.input_dim, :]), axis=0)
        
        data.z = full_z
        data.dz = full_dz
        data.ddz = full_ddz
        
        return data


    def simulate(self, z0, t):
        """
        Simulate the Rossler dynamics.

        Arguments:
            z0 - Initial condition in the form of a 3-value list or array.
            t - Array of time points at which to simulate.
            sigma, beta, rho - Lorenz parameters

        Returns:
            z, dz, ddz - Arrays of the trajectory values and their 1st and 2nd derivatives.
        """
    
        f = lambda z,t : [-z[1] -  z[2] ,
                          z[0] + self.a*z[1],
                          self.b + z[2]*(z[0] - self.c)]
        df = lambda z,dz,t : [- dz[1] - dz[2], 
                              dz[0] + self.a*dz[1],
                              z[0]*dz[2] + z[2]*dz[0]-self.c*dz[2]]
                            
        z = odeint(f, z0, t)

        dt = t[1] - t[0]
        dz = np.zeros(z.shape)
        ddz = np.zeros(z.shape)
        for i in range(t.size):
            dz[i] = f(z[i], dt*i)
            ddz[i] = df(z[i], dz[i], dt*i)
        return z, dz, ddz


    def generate_data(self, ics, t, tau=None):
        """
        Get delay embedding of first dimension

        Arguments:
            ics - Nx3 array of N initial conditions
            t - array of time points over which to simulate
            n_points - size of the high-dimensional dataset created
            linear - Boolean value. If True, high-dimensional dataset is a linear combination
            of the dynamics. If False, the dataset also includes cubic modes.
            normalization - Optional 3-value array for rescaling the 3 variables.
            sigma, beta, rho - Parameters of the Lorenz dynamics.

        Returns:
            data - Dictionary containing elements of the dataset. This includes the time points (t),
            spatial mapping (y_spatial), high-dimensional modes used to generate the full dataset
            (modes), low-dimensional  dynamics (z, along with 1st and 2nd derivatives dz and
            ddz), high-dimensional dataset (x, along with 1st and 2nd derivatives dx and ddx), and
            the true Lorenz coefficient matrix for SINDy.
        """

        n_ics = ics.shape[0]
        n_steps = t.size - self.input_dim # careful consistency 
        dt = t[1]-t[0]

        d = 3
        z = np.zeros((n_ics, t.size, d))
        dz = np.zeros(z.shape)
        ddz = np.zeros(z.shape)
        for i in range(n_ics):
            z[i], dz[i], ddz[i] = self.simulate(ics[i], t)


        if self.normalization is not None:
            z *= self.normalization
            dz *= self.normalization
            ddz *= self.normalization
 
        n = self.input_dim 
        if self.option == 'delay':
            n_delays = n
            if tau is None:
                x = np.zeros((n_ics, n_steps, n_delays))
                dx = np.zeros(x.shape)
                ddx = np.zeros(x.shape)
                for j in range(n):
                    x[:,:,j] = z[:, j:j+n_steps, 0]
                    dx[:,:,j] = dz[:, j:j+n_steps, 0]
                    ddx[:,:,j] = ddz[:, j:j+n_steps, 0]
            else:
                didx = int(tau/dt)
                n_steps = t.size - self.input_dim *didx 
                x = np.zeros((n_ics, n_steps, n_delays))
                dx = np.zeros(x.shape)
                ddx = np.zeros(x.shape)
                for j in range(n):
                    jz = j*didx
                    x[:,:,j] = z[:, jz:jz+n_steps, 0]
                    dx[:,:,j] = dz[:, jz:jz+n_steps, 0]
                    ddx[:,:,j] = ddz[:, jz:jz+n_steps, 0] 

        elif self.option == 'projection':
            raise Exception('Not Implemented')
        else:
            raise Exception('Invalid option')

        sindy_coefficients = self.sindy_coefficients()

        # Can be made a object rather than dictionary (part of class)
        data = DataStruct(name='full_sim')
        data.t = t 
        data.x = x 
        data.dx = dx
        data.ddx = ddx
        data.z = z
        data.dz = dz
        data.ddz = ddz
        data.sindy_coefficients = sindy_coefficients.astype(np.float32)
        if self.option == 'projection':
            raise Exception('Not Implemented')

        return data


    def sindy_coefficients(self):
        #TODO: Inaccurate normalization (not used)
        """
        Generate the SINDy coefficient matrix for the system.

        Arguments:
            normalization - 3-element list of array specifying scaling of each Lorenz variable
            poly_order - Polynomial order of the SINDy model.
            sigma, beta, rho - Parameters of the Lorenz system
        """
        Xi = np.zeros((library_size(3, self.poly_order, self.include_sine, True), 3))
        Xi[2,0] = -1.0 
        Xi[3,0] = -1.0 
        Xi[1,1] = 1.0 
        Xi[2,1] = self.a
        Xi[0,2] = self.b 
        Xi[3,2] = -self.c 
        Xi[6,2] = 1.0
        return Xi
