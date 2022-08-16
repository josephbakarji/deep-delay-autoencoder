from sindy_utils import library_size
import numpy as np

def get_model(name, args=None, poly_order=None, normalization=None, use_sine=False):

    if name == 'lorenz':
        args = np.array([10, 28, 8/3]) if args is None else np.array(args)
        f = lambda z, t: [args[0]*(z[1] - z[0]), 
                        z[0]*(args[1] - z[2]) - z[1], 
                        z[0]*z[1] - args[2]*z[2]]

        dim = 3
        n = normalization if normalization is not None else np.ones((dim,))
        poly_order = 2 if poly_order is None else poly_order

        Xi = np.zeros((library_size(dim, poly_order), dim))
        Xi[1,0] = - args[0] 
        Xi[2,0] = args[0]*n[0]/n[1]
        Xi[1,1] = args[1]*n[1]/n[0]
        Xi[2,1] = -1
        Xi[6,1] = -n[1]/(n[0]*n[2])
        Xi[3,2] = -args[2]
        Xi[5,2] = n[2]/(n[0]*n[1])

        z0_mean_sug = [0, 0, 25]
        z0_std_sug = [36, 48, 41]

    elif name == 'rossler':
        args = [0.2, 0.2, 5.7] if args is None else np.array(args)
        f = lambda z, t: [-z[1] -  z[2] ,
                        z[0] + args[0]*z[1],
                        args[1] + z[2]*(z[0] - args[2])]
        dim = 3
        n = normalization if normalization is not None else np.ones((dim,))
        poly_order = 2 if poly_order is None else poly_order

        Xi = np.zeros((library_size(dim, poly_order), dim))
        Xi[2,0] = -n[0]/n[1] 
        Xi[3,0] = -n[0]/n[2] 
        Xi[1,1] = n[1]/n[0] 
        Xi[2,1] = args[0] 
        Xi[0,2] = n[2]*args[1] 
        Xi[3,2] = -args[2] 
        Xi[6,2] = 1.0/n[0]

        z0_mean_sug = [0, 1, 0]
        z0_std_sug = [2, 2, 2]


    elif name == 'predator_prey':
        args = [1.0, 0.1, 1.5, 0.75] if args is None else np.array(args)
        f = lambda z, t: [args[0]*z[0] - args[1]*z[0]*z[1] ,
                        -args[2]*z[1] + args[1]*args[3]*z[0]*z[1] ]
        dim = 2
        n = normalization if normalization is not None else np.ones((dim,))
        poly_order = 2 if poly_order is None else poly_order
        Xi = np.zeros((library_size(dim, poly_order), dim))
        Xi[1,0] = args[0] 
        Xi[4,0] = -args[1] * n[0]/n[1]
        Xi[2,1] = -args[2] 
        Xi[4,1] = args[1] * args[3] * n[1]/n[0]

        z0_mean_sug = [10, 5]
        z0_std_sug = [8, 8]

    elif name == 'pendulum':
        # Not easily renormalizable because f the sin(x) feature
        # g, L,
        args = [9.8, 1] if args is None else np.array(args)
        f = lambda z, t: [z[1],
                        -args[1]/args[0]*np.sin(z[0])]
        dim = 2
        n = normalization if normalization is not None else np.ones((dim,))
        poly_order = 1 if poly_order is None else poly_order
        use_sine = True  

        Xi = np.zeros((library_size(dim, poly_order, use_sine=use_sine), dim))
        Xi[2, 0] = 1
        Xi[3, 1] = -args[1]/args[0]

        z0_mean_sug = [np.pi/2, 0]
        z0_std_sug = [np.pi/2, 2]

    return f, Xi, dim, z0_mean_sug, z0_std_sug
