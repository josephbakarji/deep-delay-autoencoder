import sys
sys.path.append("../src")
import os
from os import listdir
import shutil
import numpy as np
from training import load_model
from sindy_utils import sindy_simulate, sindy_library_names
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from lorenz import Lorenz
from waterlorenz import LorenzWW
from predprey import PredPrey
from rossler import Rossler 

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import display
import pdb
pd.options.display.float_format = '{:,.3f}'.format

def pickle2dict(params):
    params2 = {key: val[0] for key, val in params.to_dict().items()}
    list_to_int = ['input_dim', 'latent_dim', 'poly_order', 'n_ics', 'include_sine', 'exact_features']
    listwrap_to_list = ['normalization', 'system_coefficients', 'widths', 'widths_ratios']
    for key in list_to_int:
        if key in params2.keys():
            params2[key] = int(params2[key])
    for key in listwrap_to_list:
        if key in params2.keys():
            params2[key] = list(params2[key])
    return params2

def get_checkpoint_names(cpath):
    all_files = os.listdir(cpath)
    all_files = set([n.split('.')[0] for n in all_files])
    if 'checkpoint' in all_files:
        all_files.remove('checkpoint')
    all_files = list(all_files)
    all_files.sort()
    print('number of checkpoints = ', len(all_files))
    return all_files

def get_names(cases, path):
    directory = listdir(path)
    name_list = []
    for name in directory:
        for case in cases:
            if '.' not in name and case in name:
                name_list.append(name)
    name_list = list(set(name_list))

    sortidx = np.argsort(np.array([int(s.split('_')[1]) for s in name_list]))[::-1]
    name_list = [name_list[i] for i in sortidx]
    return name_list


def get_display_params(params, display_params=None):
    filt_params = dict()
    if display_params is not None:
        for key in display_params:
            if key in params.keys():
                filt_params[key] = params[key]
                print(key, ' : ', params[key])
    else:
        for key in params.keys():
            filt_params[key] = params[key]
            print(key, ' : ', params[key])
    return filt_params


def read_results(name_list, path, end_time=30, threshold=1e-2, t0_frac=0.0, end_time_plot=30, display_params=None, query_remove=False):
    ## TODO: replace by global variable DATAPATH
    varname = ['x', 'y', 'z', '1', '2']
    path = '../data/'
    known_attractor = True
    
    
    non_existing_files = []
    remove_files = []
    for name in name_list:
        print('name: ', name)
        model, params, result = load_model(name, path)
        if model is None or params is None:
            non_existing_files.append(name)
            continue
            
        params = {key: val[0] for key, val in params.to_dict().items()}
        
        end_time_idx = int(end_time_plot/params['dt'])
        
        option = params['option']
        # Backward compatibility
        if 'lorenz_coefficients' in params.keys():
            coefficients = np.array(params['lorenz_coefficients'])
        elif 'system_coefficients' in params.keys():
            coefficients = np.array(params['system_coefficients'])
        if 'model' not in params.keys():
            params['model'] = 'lorenz'
            
        if 'model' in params.keys():
            if params['model'] == 'lorenz':
                PhysicalModel = Lorenz
            elif params['model'] == 'predprey':
                PhysicalModel = PredPrey
            elif params['model'] == 'rossler':
                PhysicalModel = Rossler 
            elif params['model'] == 'lorenzww':
                PhysicalModel = LorenzWW 
            else:
                raise Exception('model doesn"t exist')
                
        if params['model'] in ['lorenzww']:
            known_attractor = False
            
        noise = params['noise']
        input_dim = int(params['input_dim'])
        latent_dim = int(params['latent_dim'])
        poly_order = int(params['poly_order'])
        IC_num = int(params['n_ics'])

        dt = params['dt']
        poly_order = int(params['poly_order'])
        include_sine = bool(params['include_sine'])

        exact_features=False
        if 'exact_features' in params.keys():
            exact_features = bool(params['exact_features'])

        coef_names = sindy_library_names(latent_dim, poly_order, include_sine=False, exact_features=exact_features)
        coef_names_full = sindy_library_names(latent_dim, poly_order, include_sine=False, exact_features=False)
        
        
        L = PhysicalModel(option=option, coefficients=coefficients, noise=noise, input_dim=input_dim, poly_order=poly_order)
        if params['model'] == 'lorenzww':
            L.filename='/home/joebakarji/delay-auto/main/examples/data/lorenzww.json'
            data = L.get_solution()
        else:
            data = L.get_solution(1, end_time, dt)

        if params['svd_dim'] is not None:
            print('Running SVD decomposition...')
            reduced_dim = int( params['svd_dim'] )
            U, s, VT = np.linalg.svd(data.x.T, full_matrices=False)
            v = np.matmul(VT[:reduced_dim, :].T, np.diag(s[:reduced_dim]))
            if params['scale']:
                scaler = StandardScaler()
                v = scaler.fit_transform(v)
            data.xorig = data.x
            data.x = v
            data.dx = np.gradient(v, params['dt'], axis=0)
            print('SVD Done!')
        
        start_time = 6
        idx = int(end_time/params['dt']) 
        idx0 = int(start_time/params['dt']) 
        test_data = [data.x[idx0:idx], data.dx[idx0:idx]]
        test_time = data.t[idx0:idx]
        if params['svd_dim'] is not None:
            test_data = [data.xorig[idx0:idx], data.x[idx0:idx], data.dx[idx0:idx]]

        prediction = model.predict(test_data)
        
        if end_time_idx > test_data[0].shape[0]:
            end_time_idx = test_data[0].shape[0] 

        ## PLOT MAIN RESULTS
        print('------- COEFFICIENTS -------')
        df = pd.DataFrame((model.sindy.coefficients).numpy()*(np.abs((model.sindy.coefficients).numpy())>threshold).astype(float), columns=varname[:latent_dim], index=coef_names)
        display(df)
        print('-------- Mask ------')
        display(pd.DataFrame(model.sindy.coefficients_mask.numpy(), columns=varname[:latent_dim], index=coef_names))
        print('-------- Parameters ------')
        params = get_display_params(params, display_params=display_params)
        
        ## PLOT LOSSES
        if result is not None:
            result_losses = result['losses'][0]
            losses_list = []
            for k in result['losses'][0].keys():
                loss = k.split('_')
                if loss[0] == 'val':
                    losses_list.append(['_'.join(loss[1:]), k])
            steps = len(result_losses['total_loss'])
            idx0 = int(t0_frac*steps)
            numrows = int(np.ceil(len(losses_list)/2))

            fig0 = plt.figure(figsize=(10, 10))
            for i, losses_couple in enumerate(losses_list):
                fig0.add_subplot(numrows, 2, i+1)
                for j, losses in enumerate(losses_couple):
                    plt.plot(range(idx0, steps), result_losses[losses][idx0:], '.-', lw=2)
                plt.title('validation_final = %.2e' % (result_losses[losses_couple[1]][-1]))
                plt.legend(losses_couple)
            
        testin = test_data[0]
        if params['svd_dim'] is not None:
            testin = test_data[1]
        
        ## PLOT PREDICTION COMPARISON 
        fig = plt.figure(figsize=(10, 3.5))
        plt.plot(test_time[:end_time_idx], testin[:end_time_idx, 0], 'b--')
        plt.plot(test_time[:end_time_idx], prediction[:end_time_idx, 0], 'r')
        plt.xlabel('t')
        plt.ylabel('x(t)')
        plt.legend(['True test', 'Auto-encoder prediction'])
        
        z_latent = model.encoder(testin).numpy()
        
        ## COMPARE RECONSTRUCTION 
#         z0 = data.z[idx0, :]
        z0 = np.array(z_latent[0])
        
        if known_attractor:
            original_sim = sindy_simulate(z0, test_time, data.sindy_coefficients, poly_order, include_sine)
            
        z_sim = sindy_simulate(z0, test_time, model.sindy.coefficients_mask* model.sindy.coefficients, poly_order, include_sine, exact_features=exact_features)
        

        if latent_dim >= 3:
            fig = plt.figure(figsize=(10, 3.5))
            ax1 = fig.add_subplot(131, projection='3d')
            ax1.plot(z_sim[:, 0], z_sim[:, 1], z_sim[:, 2], color = 'k', linewidth=1)
            ax1.set_title('Discovered SINDy dynamics')
            plt.axis('off')
            ax1.view_init(azim=120)

            if params['model'] != 'lorenzww':
                ax2 = fig.add_subplot(132, projection='3d')
                ax2.plot(original_sim[:, 0], original_sim[:, 1], original_sim[:, 2], color = 'k', linewidth=1)
                ax2.set_title('True dynamics')
                plt.xticks([])
                plt.axis('off')
                ax2.view_init(azim=120)

            ax3 = fig.add_subplot(133, projection='3d')
            ax3.plot(z_latent[:, 0], z_latent[:, 1], z_latent[:, 2], color = 'k', linewidth=1)
            ax3.set_title('Latent projection')
            plt.xticks([])
            plt.axis('off')
            ax3.view_init(azim=120)


            fig = plt.figure(figsize=(10, 3.5))
            ax1 = fig.add_subplot(131)
            ax1.plot(z_sim[:, 0], z_sim[:, 1], color = 'k', linewidth=1)
            ax1.set_label('x')
            ax1.set_label('y')
            ax1.set_title('discovered dynamics')

            ax2 = fig.add_subplot(132)
            ax2.plot(z_sim[:, 0], z_sim[:, 2], color = 'k', linewidth=1)
            ax2.set_label('x')
            ax2.set_label('z')
            ax2.set_title('discovered dynamics')

            ax3 = fig.add_subplot(133)
            ax3.plot(z_sim[:, 1], z_sim[:, 2], color = 'k', linewidth=1)
            ax3.set_label('y')
            ax3.set_label('z')
            ax3.set_title('discovered dynamics')

            fig = plt.figure(figsize=(10, 3.5))
            ax1 = fig.add_subplot(131)
            ax1.plot(z_latent[:, 0], z_latent[:, 1], color = 'k', linewidth=1)
            ax1.set_title('Latent projection')

            ax2 = fig.add_subplot(132)
            ax2.plot(z_latent[:, 0], z_latent[:, 2], color = 'k', linewidth=1)
            ax2.set_title('Latent projection')

            ax3 = fig.add_subplot(133)
            ax3.plot(z_latent[:, 1], z_latent[:, 2], color = 'k', linewidth=1)
            ax3.set_title('Latent projection')

            if known_attractor:
                fig = fig = plt.figure(figsize=(14, 10))
                ax = fig.add_subplot(311)
                ax.plot(test_time[:end_time_idx], original_sim[:end_time_idx, 0], 'b--', linewidth=2)
                ax.plot(test_time[:end_time_idx], z_latent[:end_time_idx, 0], 'r', linewidth=2)
                ax.legend(['Original', 'Latent z'])
                ax.set_ylabel('x')

                ax1 = fig.add_subplot(312)
                ax1.plot(test_time[:end_time_idx], original_sim[:end_time_idx, 1], 'b--', linewidth=2)
                ax1.plot(test_time[:end_time_idx], z_latent[:end_time_idx, 1], 'r', linewidth=2)
                ax1.legend(['Original', 'Latent z'])
                ax1.set_ylabel('y')

                ax2 = fig.add_subplot(313)
                ax2.plot(test_time[:end_time_idx] , original_sim[:end_time_idx, 2], 'b--', linewidth=2)
                ax2.plot(test_time[:end_time_idx], z_latent[:end_time_idx, 2], 'r', linewidth=2)
                ax2.legend(['Original', 'Latent z'])
                
                fig3 = plt.figure(figsize=(10, 3.5))
                plt.plot(test_time[:end_time_idx], original_sim[:end_time_idx, 0], 'b--', linewidth=2)
                plt.plot(test_time[:end_time_idx], z_sim[:end_time_idx, 0], 'r', linewidth=2)
                plt.title('Original vs. Discovered')
                plt.legend(['Original', 'Discovered'])
                ax2.set_ylabel('z')

            else:
                fig3 = plt.figure(figsize=(10, 3.5))
                plt.plot(test_time[:end_time_idx], testin[:end_time_idx, 0],  'b--', linewidth=2)
                plt.plot(test_time[:end_time_idx], z_sim[:end_time_idx, 0], 'r', linewidth=2)
                plt.title('Data vs. Discovered')
                plt.legend(['Data', 'Discovered'])

                fig3 = plt.figure(figsize=(10, 3.5))
                plt.plot(test_time[:end_time_idx], testin[:end_time_idx, 0],  'b--', linewidth=2)
                plt.plot(test_time[:end_time_idx], z_latent[:end_time_idx, 0], 'r', linewidth=2)
                plt.title('Data vs. Latent')
                plt.legend(['Data', 'Latent z'])

            plt.show()
            
        if latent_dim == 2:
            fig = plt.figure(figsize=(10, 3.5))
            ax1 = fig.add_subplot(111)
            ax1.plot(z_sim[:, 0], z_sim[:, 1], color = 'k', linewidth=1)
            ax1.set_label('x')
            ax1.set_label('y')
            ax1.set_title('discovered dynamics')

            fig = plt.figure(figsize=(10, 3.5))
            ax1 = fig.add_subplot(111)
            ax1.plot(z_latent[:, 0], z_latent[:, 1], color = 'k', linewidth=1)
            ax1.set_title('Latent projection')
            
            if known_attractor:
                fig = plt.figure(figsize=(10, 3.5))
                ax1 = fig.add_subplot(111)
                ax1.plot(original_sim[:, 0], original_sim[:, 1], color = 'k', linewidth=1)
                ax1.set_title('True dynamics')

                fig = fig = plt.figure(figsize=(14, 10))
                ax = fig.add_subplot(211)
                ax.plot(test_time, original_sim[:, 0],  'b--', linewidth=2)
                ax.plot(test_time, z_latent[:, 0], 'r', linewidth=2)
                ax.legend(['Original', 'Latent z'])
                ax.set_ylabel('x')

                ax1 = fig.add_subplot(212)
                ax1.plot(test_time, original_sim[:, 1],  'b--', linewidth=2)
                ax1.plot(test_time, z_latent[:, 1], 'r', linewidth=2)
                ax1.legend(['Original', 'Latent z'])
                ax1.set_ylabel('y')

                fig3 = plt.figure(figsize=(10, 3.5))
                plt.plot(test_time, original_sim[:, 0],  'b--', linewidth=2)
                plt.plot(test_time, z_sim[:, 0], 'r', linewidth=2)
                plt.title('Discovered vs. True dynamics')
                plt.legend(['Original', 'Discovered'])

            plt.show()

        if query_remove:
            print('Do you want to remove this file? Y/N')
            answer = input()
            if answer == 'Y' or answer == 'y':
                remove_files.append(name)

    return non_existing_files, remove_files


def delete_results(file_list, path):
    dir_files = listdir(path)
    for fdir in dir_files:
        for fdelete in file_list:
            if fdelete in fdir:
                print('deleting: ', fdir)
                if os.path.isdir(path+fdir):
                    shutil.rmtree(path+fdir)
                else:
                    os.remove(path+fdir)
