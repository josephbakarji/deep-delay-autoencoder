import numpy as np
from scipy.integrate import odeint
from scipy.special import legendre, chebyt
import sys
sys.path.append('../src')
from sindy_utils import library_size
from data_manage import DataStruct
import pdb


class Lorenz:
    def __init__(self, 
            option='delay', 
            coefficients=[10, 8/3, 28.], 
            noise=0.0, 
            input_dim=128,
            normalization=[1/40, 1/40, 1/40], 
            linear=False,
            poly_order=3):

        self.option = option
        self.sigma = coefficients[0]
        self.beta = coefficients[1]
        self.rho = coefficients[2]
        self.noise = noise
        self.input_dim = input_dim
        self.normalization = np.array(normalization) if normalization is not None else np.array([1, 1, 1])
        self.linear = linear
        self.poly_order = poly_order

    
    def get_solution(self, n_ics, tend, dt, ic_means=[0, 0, 25], ic_widths=[36, 48, 41], tau=None):
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



    def simulate_lorenz(self, z0, t):
        """
        Simulate the Lorenz dynamics.

        Arguments:
            z0 - Initial condition in the form of a 3-value list or array.
            t - Array of time points at which to simulate.
            sigma, beta, rho - Lorenz parameters

        Returns:
            z, dz, ddz - Arrays of the trajectory values and their 1st and 2nd derivatives.
        """
        f = lambda z,t : [self.sigma*(z[1] - z[0]), z[0]*(self.rho - z[2]) - z[1], z[0]*z[1] - self.beta*z[2]]
        df = lambda z,dz,t : [self.sigma*(dz[1] - dz[0]),
                            dz[0]*(self.rho - z[2]) + z[0]*(-dz[2]) - dz[1],
                            dz[0]*z[1] + z[0]*dz[1] - self.beta*dz[2]]

        z = odeint(f, z0, t)

        dt = t[1] - t[0]
        dz = np.zeros(z.shape)
        ddz = np.zeros(z.shape)
        for i in range(t.size):
            dz[i] = f(z[i],dt*i)
            ddz[i] = df(z[i], dz[i], dt*i)
        return z, dz, ddz


    def generate_data(self, ics, t, tau=None):
        """
        Generate high-dimensional Lorenz data set.

        Arguments:
            ics - Nx3 array of N initial conditions
            t - array of time points over which to simulate
            n_points - size of the high-dimensional dataset created
            linear - Boolean value. If True, high-dimensional dataset is a linear combination
            of the Lorenz dynamics. If False, the dataset also includes cubic modes.
            normalization - Optional 3-value array for rescaling the 3 Lorenz variables.
            sigma, beta, rho - Parameters of the Lorenz dynamics.

        Returns:
            data - Dictionary containing elements of the dataset. This includes the time points (t),
            spatial mapping (y_spatial), high-dimensional modes used to generate the full dataset
            (modes), low-dimensional Lorenz dynamics (z, along with 1st and 2nd derivatives dz and
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
            z[i], dz[i], ddz[i] = self.simulate_lorenz(ics[i], t)


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
                    x[:, :, j] = z[:, j:j+n_steps, 0]
                    dx[:, :, j] = dz[:, j:j+n_steps, 0]
                    ddx[:, :, j] = ddz[:, j:j+n_steps, 0]
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
            L = 1
            y_spatial = np.linspace(-L, L, n)

            modes = np.zeros((2*d, n))
            for i in range(2*d):
                modes[i] = legendre(i)(y_spatial)
                # modes[i] = chebyt(i)(y_spatial)
                # modes[i] = np.cos((i+1)*np.pi*y_spatial/2)
            x1 = np.zeros((n_ics,n_steps,n))
            x2 = np.zeros((n_ics,n_steps,n))
            x3 = np.zeros((n_ics,n_steps,n))
            x4 = np.zeros((n_ics,n_steps,n))
            x5 = np.zeros((n_ics,n_steps,n))
            x6 = np.zeros((n_ics,n_steps,n))

            x = np.zeros((n_ics,n_steps,n))
            dx = np.zeros(x.shape)
            ddx = np.zeros(x.shape)
            for i in range(n_ics):
                for j in range(n_steps):
                    x1[i,j] = modes[0]*z[i,j,0]
                    x2[i,j] = modes[1]*z[i,j,1]
                    x3[i,j] = modes[2]*z[i,j,2]
                    x4[i,j] = modes[3]*z[i,j,0]**3
                    x5[i,j] = modes[4]*z[i,j,1]**3
                    x6[i,j] = modes[5]*z[i,j,2]**3

                    x[i,j] = x1[i,j] + x2[i,j] + x3[i,j]
                    if not self.linear:
                        x[i,j] += x4[i,j] + x5[i,j] + x6[i,j]

                    dx[i,j] = modes[0]*dz[i,j,0] + modes[1]*dz[i,j,1] + modes[2]*dz[i,j,2]
                    if not self.linear:
                        dx[i,j] += modes[3]*3*(z[i,j,0]**2)*dz[i,j,0] + modes[4]*3*(z[i,j,1]**2)*dz[i,j,1] + modes[5]*3*(z[i,j,2]**2)*dz[i,j,2]
                    
                    ddx[i,j] = modes[0]*ddz[i,j,0] + modes[1]*ddz[i,j,1] + modes[2]*ddz[i,j,2]
                    if not self.linear:
                        ddx[i,j] += modes[3]*(6*z[i,j,0]*dz[i,j,0]**2 + 3*(z[i,j,0]**2)*ddz[i,j,0]) \
                                + modes[4]*(6*z[i,j,1]*dz[i,j,1]**2 + 3*(z[i,j,1]**2)*ddz[i,j,1]) \
                                + modes[5]*(6*z[i,j,2]*dz[i,j,2]**2 + 3*(z[i,j,2]**2)*ddz[i,j,2])
        else:
            raise Exception('Invalid option')

        sindy_coefficients = self.lorenz_coefficients()

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




#################################
################################
################################
###### FROM example_lorenz_delay.py (NOT FIXED) #########3
# def get_lorenz_data_withDelaysAsz(n_training_ics, n_validation_ics, n_test_ics, n_delays):
#     t = np.arange(0, 10, .002)
#     n_steps = t.size - n_delays
#     N = n_delays
    
#     ic_means = np.array([0,0,25])
#     ic_widths = 2*np.array([36,48,41])
#     d = 3

#     noise_strength = 0

#     # training data
#     ics = ic_widths*(np.random.rand(n_training_ics, 3)-.5) + ic_means
#     training_data = generate_lorenz_data(ics, t, N, normalization=np.array([1/40,1/40,1/40]))
#     training_data['x'] = training_data['x'].reshape((-1,N)) + noise_strength*np.random.randn(n_steps*n_training_ics,N)
#     training_data['dx'] = training_data['dx'].reshape((-1,N)) + noise_strength*np.random.randn(n_steps*n_training_ics,N)
#     training_data['ddx'] = training_data['ddx'].reshape((-1,N)) + noise_strength*np.random.randn(n_steps*n_training_ics,N)

#     U,s,V = np.linalg.svd(training_data['x'], full_matrices=False)
#     training_data['z'] = U[:,0:d]

#     # validation data
#     ics = ic_widths*(np.random.rand(n_validation_ics, 3)-.5) + ic_means
#     validation_data = generate_lorenz_data(ics, t, N, normalization=np.array([1/40,1/40,1/40]))
#     validation_data['x'] = validation_data['x'].reshape((-1,N)) + noise_strength*np.random.randn(n_steps*n_validation_ics,N)
#     validation_data['dx'] = validation_data['dx'].reshape((-1,N)) + noise_strength*np.random.randn(n_steps*n_validation_ics,N)
#     validation_data['ddx'] = validation_data['ddx'].reshape((-1,N)) + noise_strength*np.random.randn(n_steps*n_validation_ics,N)
    
#     validation_data['z'] = (np.dot(validation_data['x'], V.T)/s)[:,0:d]

#     # test data
#     ics = ic_widths*(np.random.rand(n_test_ics, 3)-.5) + ic_means
#     test_data = generate_lorenz_data(ics, t, N, normalization=np.array([1/40,1/40,1/40]))
#     test_data['x'] = test_data['x'].reshape((-1,N)) + noise_strength*np.random.randn(n_steps*n_test_ics,N)
#     test_data['dx'] = test_data['dx'].reshape((-1,N)) + noise_strength*np.random.randn(n_steps*n_test_ics,N)
#     test_data['ddx'] = test_data['ddx'].reshape((-1,N)) + noise_strength*np.random.randn(n_steps*n_test_ics,N)
    
#     test_data['z'] = (np.dot(test_data['x'], V.T)/s)[:,0:d]

#     return training_data, validation_data, test_data
