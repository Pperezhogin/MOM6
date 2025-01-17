import sys
import numpy as np
import xarray as xr
from helpers.cm26 import read_datasets, propagate_mask
from helpers.ann_tools import ANN, export_ANN, tensor_from_xarray, torch_pad
import torch
import torch.optim as optim
import json
import itertools

import os
from time import time

def fetch_data(dataset, factor=None, depth=None, ds_str='train'):
    '''
    Here dataset is the output of read_datasets, i.e.
    dictionary of cm26 datasets
    '''
    ds = dataset[f'{ds_str}-{factor}']
    time_random = np.random.randint(len(ds.data.time))
    # Here we explicitly remove the two upmost grid points near the Polar Fold
    # Fluxes and B.C. there are not well defined
    data_xarray = ds.data.isel(time=time_random, zl=depth).isel(yh=slice(None,-2))

    data = {}
    for key in ['Txx', 'Txy', 'Tyy', 'sh_xx', 'sh_xy_h', 'rel_vort_h', 'wet']:
        data[key] = tensor_from_xarray(data_xarray[key])
    data['areaT'] = tensor_from_xarray(data_xarray.delta_x)**2

    del data_xarray

    iT_norm = 1. / torch.sqrt((data['Txx']**2 + data['Tyy']**2 + data['Txy']**2).mean())
    for key in ['Txx', 'Txy', 'Tyy']:
        data[key] *= iT_norm

    return ds, data, iT_norm

def train_ANN_fluxes(factors=[12,15],
              stencil_size = 3,
              hidden_layers=[20],
              symmetries='False',
              time_iters=10,
              learning_rate = 1e-3,
              depth_idx=np.arange(1),
              print_iters=1,
              gradient_features=['sh_xy', 'sh_xx', 'rel_vort'],
              permute_factors_and_depth=True,
              subfilter='subfilter-large',
              FGR=3):
    '''
    time_iters is the number of time snaphots
    randomly sampled for each factor and depth

    depth_idx is the indices of the vertical layers which
    participate in training process
    '''
    ########### Read dataset ############
    dataset = read_datasets(['train', 'validate'], factors, subfilter=subfilter, FGR=FGR)

    ########## Init logger ###########
    logger = xr.Dataset()
    for key in ['MSE_train', 'MSE_validate']:
        logger[key] = xr.DataArray(np.zeros([time_iters, len(factors), len(depth_idx)]), 
                                   dims=['iter', 'factor', 'depth'], 
                                   coords={'factor': factors, 'depth': depth_idx})

    ########## Init ANN ##############
    # As default we have 3 input features on a stencil: D, D_hat and vorticity
    num_input_features = stencil_size**2 * len(gradient_features)
    ann_Tall = ANN([num_input_features, *hidden_layers, 3])
    
    ########## Symmetries as data augmentation ######
    def augment():
        if symmetries == 'False':
            # Values for rotation, reflect_x and reflect_y
            return zip([0],[False],[False])
        elif symmetries == 'True':
            rots  = [90, 0]
            refxs = [True, False]
            refys = [True, False]
            return zip([rots[np.random.binomial(1,0.5)]], 
                        [refxs[np.random.binomial(1,0.5)]], 
                        [refys[np.random.binomial(1,0.5)]])
        elif symmetries == 'All':
            rots  = [90, 0]
            refxs = [True, False]
            refys = [True, False]
            return itertools.product(rots, refxs, refys)
        elif symmetries == 'None':
            rots  = [0, 0]
            refxs = [False, False]
            refys = [False, False]
            return itertools.product(rots, refxs, refys)
        else:
            print('Wrong symmetries parameter:', symmetries)

    ########## Random sampling of depth and factors #######
    def iterator(x,y):
        # Product of two 1D iterators
        x_prod = np.repeat(x,len(y))
        y_prod = np.tile(y,len(x))
        xy_prod = np.vstack([x_prod,y_prod]).T
        if permute_factors_and_depth:
            # Randomly permuting iterator along common dimension
            return np.random.permutation(xy_prod)
        else:
            # This is equivalent to
            # for xx in x:
            #    for yy in y:
            #       ....
            return xy_prod
    
    ############ Init optimizer ##############
    
    all_parameters = ann_Tall.parameters()
    
    optimizer = optim.Adam(all_parameters, lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
            milestones=[int(time_iters/2), int(time_iters*3/4), int(time_iters*7/8)], gamma=0.1)

    t_s = time()
    for time_iter in range(time_iters):
        t_e = time()

        for factor, depth in iterator(factors, depth_idx):
            ds, data, iT_norm = fetch_data(dataset, factor, depth, 'train')
            
            ######## Optionally, apply symmetries by data augmentation #########
            for rotation, reflect_x, reflect_y in augment():
                optimizer.zero_grad()

                prediction = ds.state.ANN_inference(ann_Tall = ann_Tall, stencil_size=stencil_size,
                                 gradient_features=gradient_features,
                                 rotation=rotation, reflect_x=reflect_x, reflect_y=reflect_y,
                                 data=data)
                
                for key in ['Txx', 'Txy', 'Tyy']:
                    prediction[key] *= iT_norm

                MSE_train = (
                    (prediction['Txx'] - data['Txx'])**2 + 
                    (prediction['Txy'] - data['Txy'])**2 + 
                    (prediction['Tyy'] - data['Tyy'])**2 ).mean()
                
                MSE_train.backward()
                optimizer.step()

            del ds

            ############ Validation step ##################
            with torch.no_grad():
                ds, data, iT_norm = fetch_data(dataset, factor, depth, 'validate')
                prediction = ds.state.ANN_inference(ann_Tall = ann_Tall, stencil_size=stencil_size,
                                 gradient_features=gradient_features,
                                 rotation=rotation, reflect_x=reflect_x, reflect_y=reflect_y,
                                 data=data)
                
                for key in ['Txx', 'Txy', 'Tyy']:
                    prediction[key] *= iT_norm

                MSE_validate = (
                    (prediction['Txx'] - data['Txx'])**2 + 
                    (prediction['Txy'] - data['Txy'])**2 + 
                    (prediction['Tyy'] - data['Tyy'])**2 ).mean()

            del ds
        
            ########### Logging ############
            MSE_train = float(MSE_train.data)
            MSE_validate = float(MSE_validate.data)

            for key in ['MSE_train', 'MSE_validate']:
                logger[key].loc[{'iter': time_iter, 'factor': factor, 'depth': depth}] = eval(key)
            if (time_iter+1) % print_iters == 0:
                print(f'Factor: {factor}, depth: {depth}, '+'MSE train/validate: [%.6f, %.6f]' % (MSE_train, MSE_validate))
        t = time()
        if (time_iter+1) % print_iters == 0:
            print(f'Iter/num_iters [{time_iter+1}/{time_iters}]. Iter time/Remaining time in seconds: [%.2f/%.1f]' % (t-t_e, (t-t_s)*(time_iters/(time_iter+1)-1)))
        scheduler.step()

    for factor in factors:
        for train_str in ['train', 'validate']:
            del dataset[f'{train_str}-{factor}']
    
    return ann_Tall, logger