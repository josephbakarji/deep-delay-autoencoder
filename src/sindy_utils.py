import numpy as np
from scipy.special import binom
from scipy.integrate import odeint
import pdb


def library_size(n, poly_order, use_sine=False, include_constant=True):
    l = 0
    for k in range(poly_order+1):
        l += int(binom(n+k-1,k))
    if use_sine:
        l += n
    if not include_constant:
        l -= 1
    return l


def sindy_library(X, poly_order, include_sine=False, include_names=False, exact_features=False, include_sparse_weighting=False):
    # Upgrade to combinations
    symbs = ['x', 'y', 'z', '4', '5', '6', '7']

    # GENERALIZE 
    if exact_features:
        # Check size (is first dimension batch?)
        library = np.ones((X.shape[0],5))
        library[:, 0] = X[:,0]
        library[:, 1] = X[:,1]
        library[:, 2] = X[:,2]
        library[:, 3] = X[:,0]* X[:,1]
        library[:, 4] = X[:,0]* X[:,2]
        names = ['x', 'y', 'z', 'xy', 'xz']
    else:
        
        m, n = X.shape
        l = library_size(n, poly_order, include_sine, True)
        library = np.ones((m, l))
        sparse_weights = np.ones((l, n))
        
        index = 1
        names = ['1']

        for i in range(n):
            library[:,index] = X[:,i]
            sparse_weights[index, :] *= 1
            names.append(symbs[i])
            index += 1

        if poly_order > 1:
            for i in range(n):
                for j in range(i,n):
                    library[:,index] = X[:,i]*X[:,j]
                    sparse_weights[index, :] *= 2
                    names.append(symbs[i]+symbs[j])
                    index += 1

        if poly_order > 2:
            for i in range(n):
                for j in range(i,n):
                    for k in range(j,n):
                        library[:,index] = X[:,i]*X[:,j]*X[:,k]
                        sparse_weights[index, :] *= 3
                        names.append(symbs[i]+symbs[j]+symbs[k])
                        index += 1

        if poly_order > 3:
            for i in range(n):
                for j in range(i,n):
                    for k in range(j,n):
                        for q in range(k,n):
                            library[:,index] = X[:,i]*X[:,j]*X[:,k]*X[:,q]
                            sparse_weights[index, :] *= 4
                            names.append(symbs[i]+symbs[j]+symbs[k]+symbs[q])
                            index += 1

        if poly_order > 4:
            for i in range(n):
                for j in range(i,n):
                    for k in range(j,n):
                        for q in range(k,n):
                            for r in range(q,n):
                                library[:,index] = X[:,i]*X[:,j]*X[:,k]*X[:,q]*X[:,r]
                                sparse_weights[index, :] *= 5
                                names.append(symbs[i]+symbs[j]+symbs[k]+symbs[q]+symbs[r])
                                index += 1

        if include_sine:
            for i in range(n):
                library[:,index] = np.sin(X[:,i])
                names.append('sin('+symbs[i]+')')
                index += 1

       
    return_list = [library]
    if include_names:
        return_list.append(names)
    if include_sparse_weighting:
        return_list.append(sparse_weights)
    return return_list

def sindy_library_names(latent_dim, poly_order, include_sine=False, exact_features=False):
    # Upgrade to combinations
    symbs = ['x', 'y', 'z', '4', '5', '6', '7']
    
    if exact_features:
        return ['x', 'y', 'z', 'xy', 'xz']
    
    n = latent_dim
    index = 1
    names = ['1']

    for i in range(n):
        names.append(symbs[i])
        index += 1

    if poly_order > 1:
        for i in range(n):
            for j in range(i,n):
                names.append(symbs[i]+symbs[j])
                index += 1

    if poly_order > 2:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    names.append(symbs[i]+symbs[j]+symbs[k])
                    index += 1

    if poly_order > 3:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    for q in range(k,n):
                        names.append(symbs[i]+symbs[j]+symbs[k]+symbs[q])
                        index += 1
                    
    if poly_order > 4:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    for q in range(k,n):
                        for r in range(q,n):
                            names.append(symbs[i]+symbs[j]+symbs[k]+symbs[q]+symbs[r])
                            index += 1

    if include_sine:
        for i in range(n):
            names.append('sin('+symbs[i]+')')
            index += 1
    
    return names


def sindy_library_order2(X, dX, poly_order, include_sine=False):
    m,n = X.shape
    l = library_size(2*n, poly_order, include_sine, True)
    library = np.ones((m,l))
    index = 1

    X_combined = np.concatenate((X, dX), axis=1)

    for i in range(2*n):
        library[:,index] = X_combined[:,i]
        index += 1

    if poly_order > 1:
        for i in range(2*n):
            for j in range(i,2*n):
                library[:,index] = X_combined[:,i]*X_combined[:,j]
                index += 1

    if poly_order > 2:
        for i in range(2*n):
            for j in range(i,2*n):
                for k in range(j,2*n):
                    library[:,index] = X_combined[:,i]*X_combined[:,j]*X_combined[:,k]
                    index += 1

    if poly_order > 3:
        for i in range(2*n):
            for j in range(i,2*n):
                for k in range(j,2*n):
                    for q in range(k,2*n):
                        library[:,index] = X_combined[:,i]*X_combined[:,j]*X_combined[:,k]*X_combined[:,q]
                        index += 1
                    
    if poly_order > 4:
        for i in range(2*n):
            for j in range(i,2*n):
                for k in range(j,2*n):
                    for q in range(k,2*n):
                        for r in range(q,2*n):
                            library[:,index] = X_combined[:,i]*X_combined[:,j]*X_combined[:,k]*X_combined[:,q]*X_combined[:,r]
                            index += 1

    if include_sine:
        for i in range(2*n):
            library[:,index] = np.sin(X_combined[:,i])
            index += 1


def sindy_fit(RHS, LHS, coefficient_threshold):
    m,n = LHS.shape
    Xi = np.linalg.lstsq(RHS,LHS, rcond=None)[0]
    
    for k in range(10):
        small_inds = (np.abs(Xi) < coefficient_threshold)
        Xi[small_inds] = 0
        for i in range(n):
            big_inds = ~small_inds[:,i]
            if np.where(big_inds)[0].size == 0:
                continue
            Xi[big_inds,i] = np.linalg.lstsq(RHS[:,big_inds], LHS[:,i], rcond=None)[0]
    return Xi


def sindy_simulate(x0, t, Xi, poly_order, include_sine=False, exact_features=False):
    m = t.size
    n = x0.size
    f = lambda x,t : np.dot(sindy_library(np.array(x).reshape((1,n)), poly_order, include_sine, include_names=False, exact_features=exact_features), Xi).reshape((n,))
    x = odeint(f, x0, t)
    return x


def sindy_simulate_order2(x0, dx0, t, Xi, poly_order, include_sine):
    m = t.size
    n = 2*x0.size
    l = Xi.shape[0]

    Xi_order1 = np.zeros((l,n))
    for i in range(n//2):
        Xi_order1[2*(i+1),i] = 1.
        Xi_order1[:,i+n//2] = Xi[:,i]
    
    x = sindy_simulate(np.concatenate((x0,dx0)), t, Xi_order1, poly_order, include_sine)
    return x
