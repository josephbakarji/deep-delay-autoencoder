import sys
sys.path.append("../src")
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from os import listdir
import shutil
import numpy as np
from sindy_utils import sindy_simulate, sindy_library_names
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from solvers import SynthData

import pickle5 as pickle
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import display
import pdb
pd.options.display.float_format = '{:,.3f}'.format


def params_names():
    my_params = ['loss_weight_integral', 'sindy_pert', 'svd_dim', 'model']

    primary_params = ['case', 'coefficient_initialization', 'exact_features', 'fix_coefs', 'input_dim', 'latent_dim', 
                     'loss_weight_integral', 'loss_weight_rec', 'loss_weight_sindy_regularization', 'loss_weight_sindy_x', 
                     'loss_weight_sindy_z', 'loss_weight_x0', 'model', 'n_ics', 'widths_ratios', 'svd_dim']
    secondary_params = ['activation', 'actual_coefficients', 'coefficient_threshold', 'dt', 
                        'fixed_coefficient_mask', 'library_dim',
                       'max_epochs', 'model_order', 'noise', 'option', 'patience', 'poly_order', 'print_frequency', 
                        'save_checkpoints', 'save_freq', 'scale', 'sindy_pert']
    tertiary_params = ['batch_size', 'data_path', 'include_sine', 'learning_rate', 'learning_rate_sched', 'print_progress']
    
    return primary_params, secondary_params, tertiary_params


def pickle2dict(params):
    params2 = {key: val[0] for key, val in params.to_dict().items()}
    list_to_int = ['input_dim', 'latent_dim', 'poly_order', 'n_ics', 'include_sine', 'exact_features']
    listwrap_to_list = ['normalization', 'system_coefficients', 'widths', 'widths_ratios']
    for key in list_to_int:
        if key in params2.keys():
            params2[key] = int(params2[key])
    for key in listwrap_to_list:
        if key in params2.keys():
            if params2[key] is not None:
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



def get_cases(path, filter_case=None, print_cases=True):
    directory = listdir(path)
    case_list = []
    for name in directory:
        if '.' not in name:
            casename = '_'.join(name.split('_')[2:])
            if filter_case is not None:
                if fcase in name:
                    for fcase in filter_case:
                        case_list.append(casename)
            else:
                case_list.append(casename)
    case_list = list(set(case_list))
    if print_cases:
        for case in case_list: 
            print(case)
    return case_list


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


def load_results(name, path='./results/'):
    try:
        model = tf.keras.models.load_model(path+name)
    except:
        print('model file doesnt exist')
        model = None
        
    try:
        params = pickle.load(open(path+name+'_params.pkl', 'rb'))
    except:
        print('params file doesnt not exist')
        params = None
    
    try:
        results = pickle.load(open(path+name+'_results.pkl', 'rb')) 
    except:
        results = None
        print('no results file for ', name)

    return model, params, results

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
        
def make_inputs_svd(S, reduced_dim, scale):
    if reduced_dim is not None:
        print('Running SVD...')
        U, s, VT = np.linalg.svd(S.x.T, full_matrices=False)
        v = np.matmul(VT[:reduced_dim, :].T, np.diag(s[:reduced_dim]))
        if params['scale']:
            scaler = StandardScaler()
            v = scaler.fit_transform(v)
        S.xorig = S.x
        S.x = v
        S.dx = np.gradient(v, params['dt'], axis=0)
        print('SVD Done!')
    return S


def read_results(name_list, 
                    path, 
                    start_time=6, 
                    end_time=30, 
                    threshold=1e-2, 
                    t0_frac=0.0, 
                    end_time_plot=30, 
                    display_params=None, 
                    query_remove=False, 
                    known_attractor=False):

    varname = list('xyz123')
    known_attractor = True
    non_existing_models = []
    non_existing_params = []
    remove_files = []
    
    for name in name_list:
        print('name: ', name)
        model, params, result = load_results(name, path)
        print('got results...')
        if model is None or params is None:
            if model is None: non_existing_models.append(name)
            if params is None: non_existing_params.append(name)
            continue
        
        params = pickle2dict(params)
        end_time_idx = int(end_time_plot/params['dt'])
        
        # FIX Experimental data
        S = SynthData(model=params['model'], 
                args=params['system_coefficients'], 
                noise=params['noise'], 
                input_dim=params['input_dim'], 
                normalization=params['normalization'])
        print('Generating Test Solution...')
        S.run_sim(1, end_time, params['dt'])
        
#         if params['model'] == 'lorenzww':
#             L.filename='/home/joebakarji/delay-auto/main/examples/data/lorenzww.json'
#             data = L.get_solution()
#         else:
#             data = L.get_solution(1, end_time, params['dt'])

        ## Get SVD data (write in separate function)
        S = make_inputs_svd(S, params['svd_dim'], params['scale'])
            
        ## This seems arbitrary
        idx = int(end_time/params['dt']) 
        idx0 = int(start_time/params['dt']) 
        test_data = [S.x[:, idx0:idx].T, S.dx[:, idx0:idx].T]
        test_time = S.t[idx0:idx]
        if params['svd_dim'] is not None:
            test_data = [S.xorig[:, idx0:idx].T] +  test_data

        prediction = model.predict(test_data)
        
        if end_time_idx > test_data[0].shape[0]:
            end_time_idx = test_data[0].shape[0] 

        ## Display Optimal Coefficients 
        coef_names = sindy_library_names(params['latent_dim'], 
                                         params['poly_order'], 
                                         include_sine=params['include_sine'], 
                                         exact_features=params['exact_features'])

        print('------- COEFFICIENTS -------')
        df = pd.DataFrame((model.sindy.coefficients).numpy()*
                          (np.abs((model.sindy.coefficients).numpy()) > threshold).astype(float), 
                          columns=varname[:params['latent_dim']], 
                          index=coef_names)
        display(df)

        print('-------- Mask ------')
        display(pd.DataFrame(model.sindy.coefficients_mask.numpy(), columns=varname[:params['latent_dim']], index=coef_names))

        print('-------- Parameters ------')
        disp_params = get_display_params(params, display_params=display_params)
        


        ## PLOT LOSSES
        plot_losses(result, t0_frac=t0_frac)
            
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
        z0 = np.array(z_latent[0])
        z_sim = sindy_simulate(z0, test_time, model.sindy.coefficients_mask* model.sindy.coefficients, 
                                params['poly_order'], params['include_sine'], exact_features=params['exact_features'])
        
        if known_attractor:
            original_sim = sindy_simulate(z0, test_time, S.sindy_coefficients, params['poly_order'], params['include_sine'])
        else:
            original_sim = None
            

        ## PLOT RESULTS
        # Assuming n=2 or n=3
        plot_portraits(z_sim, n=params['latent_dim'], title='Discovered Simulated Dynamics')
        plot_portraits(z_latent, n=params['latent_dim'], title='Latent Variable')

        plot_txy(test_time[:end_time_idx], testin[:end_time_idx, :], z_sim[:end_time_idx, :], n=1,
                title='Input vs. Discovered 1st Dim.', names=['Input data', 'Discovered']) 
        plot_txy(test_time[:end_time_idx], testin[:end_time_idx, :], z_latent[:end_time_idx, :], n=1,
                title='Input vs. Latent z_0', names=['Input data', 'Latent']) 

        if params['latent_dim'] > 2:
            plot3d_comparison(z_sim, z_latent, zorig=original_sim, title='Discovered SINDy dynamics')

        if known_attractor:
            plot_txy(test_time[:end_time_idx], original_sim[:end_time_idx, :], z_latent[:end_time_idx, :], 
                    n=z_latent.shape[1], names=['Original', 'Latent'], title='')
            plot_txy(test_time[:end_time_idx], original_sim[:end_time_idx, :], z_sim[:end_time_idx, :],
                    n=z_latent.shape[1], names=['Original', 'Discovered'], title='')

        plt.show()

        ## Data cleaing while going through results
        if query_remove:
            print('Do you want to remove this file? Y/N')
            answer = input()
            if answer == 'Y' or answer == 'y':
                remove_files.append(name)

    return non_existing_models, non_existing_params, remove_files


###### PLOT FUNCTIONS ########

def plot_txy(t, x, y, n=1, names=['x', 'y'], title=''):
    fig = plt.figure(figsize=(12, 4))
    ax = []
    for i in range(n):
        axp = fig.add_subplot(n, 1, i+1)
        ax.append( axp )
        ax[i].plot(t, x[:, i], 'b--', linewidth=2)
        ax[i].plot(t, y[:, i], 'r', linewidth=2)
        ax[i].legend([names[0], names[1]])
        ax[i].set_ylabel('x_'+str(i))
    return ax
    
def plot_portraits(z_sim, n=2, title=''):
    if n==2:
        n_figs = 1
        figwidth = 3.5
    elif n==3:
        n_figs = 3
        figwidth=10

    fig = plt.figure(figsize=(figwidth, 3.5))
    ax1 = fig.add_subplot(1, n_figs, 1)

    ax1.plot(z_sim[:, 0], z_sim[:, 1], color = 'k', linewidth=1)
    ax1.set_label('x')
    ax1.set_label('y')
    ax1.set_title(title)

    if n==3:
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.plot(z_sim[:, 0], z_sim[:, 2], color = 'k', linewidth=1)
        ax2.set_label('x')
        ax2.set_label('z')
        ax2.set_title(title)

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.plot(z_sim[:, 1], z_sim[:, 2], color = 'k', linewidth=1)
        ax3.set_label('y')
        ax3.set_label('z')
        ax3.set_title(title)



def plot3d_comparison(zsim, zlatent, zorig=None, title=''):
    fig = plt.figure(figsize=(10, 3.5))
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(zsim[:, 0], zsim[:, 1], zsim[:, 2], color = 'k', linewidth=1)
    ax1.set_title(title)
    plt.axis('off')
    ax1.view_init(azim=120)

    ax3 = fig.add_subplot(132, projection='3d')
    ax3.plot(zlatent[:, 0], zlatent[:, 1], zlatent[:, 2], color = 'k', linewidth=1)
    ax3.set_title('Latent projection')
    plt.xticks([])
    plt.axis('off')
    ax3.view_init(azim=120)

    if zorig is not None:
        ax2 = fig.add_subplot(133, projection='3d')
        ax2.plot(zorig[:, 0], zorig[:, 1], zorig[:, 2], color = 'k', linewidth=1)
        ax2.set_title('True dynamics')
        plt.xticks([])
        plt.axis('off')
        ax2.view_init(azim=120)


def plot_losses(result, t0_frac=0.0):
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
    else:
        print("NO LOSSES RESULTS FILE")
