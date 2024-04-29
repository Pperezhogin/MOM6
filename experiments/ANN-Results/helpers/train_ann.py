import sys
import numpy as np
import xarray as xr
from helpers.cm26 import read_datasets
from helpers.ann_tools import ANN, export_ANN, tensor_from_xarray
import torch
import torch.optim as optim
import json
import itertools

import os
from time import time

def get_SGS(batch):
    SGSx = tensor_from_xarray(batch.data.SGSx)
    SGSy = tensor_from_xarray(batch.data.SGSy)

    SGS_norm = 1. / torch.sqrt((SGSx**2 + SGSy**2).mean())
    SGSx = SGSx * SGS_norm
    SGSy = SGSy * SGS_norm

    return SGSx, SGSy, SGS_norm

def MSE(batch, SGSx, SGSy, SGS_norm, ann_Txy, ann_Txx_Tyy, rotation=0, reflect_x=False, reflect_y=False):
    prediction = batch.state.Apply_ANN(ann_Txy, ann_Txx_Tyy,
        rotation=rotation, reflect_x=reflect_x, reflect_y=reflect_y)

    ANNx = prediction['ZB20u'] * SGS_norm
    ANNy = prediction['ZB20v'] * SGS_norm
    MSE_train = ((ANNx-SGSx)**2 + (ANNy-SGSy)**2).mean()
    return MSE_train

def train_ANN(factors=[12,15],
              stencil_size = 3,
              hidden_layers=[20], 
              symmetries=False,
              time_iters=10,
              depth_idx=np.arange(1),
              print_iters=1):
    '''
    time_iters is the number of time snaphots
    randomly sampled for each factor and depth

    depth_idx is the indices of the vertical layers which
    participate in training process
    '''
    ########### Read dataset ############
    dataset = read_datasets(['train', 'validate'], factors)

    ########## Init logger ###########
    logger = xr.Dataset()
    for key in ['MSE_train', 'MSE_validate']:
        logger[key] = xr.DataArray(np.zeros([time_iters, len(factors), len(depth_idx)]), 
                                   dims=['iter', 'factor', 'depth'], 
                                   coords={'factor': factors})

    ########## Init ANN ##############
    # As default we have 3 input features on a stencil: D, D_hat and vorticity
    num_input_features = stencil_size**2 * 3
    ann_Txy = ANN([num_input_features, *hidden_layers, 1])
    ann_Txx_Tyy = ANN([num_input_features, *hidden_layers, 2])
    
    ########## Symmetries as data augmentation ######
    def augment():
        kw = {}
        if symmetries:
            # Admissible values
            rots  = [90, 0]
            refxs = [True, False]
            refys = [True, False]
            # Sampling
            kw['rotation'] = rots[np.random.binomial(1,0.5)]
            kw['reflect_x'] = refxs[np.random.binomial(1,0.5)]
            kw['reflect_y'] = refys[np.random.binomial(1,0.5)]
        else:
            kw['rotation'] = 0
            kw['reflect_x'] = False
            kw['reflect_y'] = False
        return kw

    ############ Init optimizer ##############
    all_parameters = list(ann_Txy.parameters()) + list(ann_Txx_Tyy.parameters())
    optimizer = optim.Adam(all_parameters, lr=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
            milestones=[int(time_iters/2), int(time_iters*3/4), int(time_iters*7/8)], gamma=0.1)

    t_s = time()
    for time_iter in range(time_iters):
        t_e = time()

        for ifactor, factor in enumerate(factors):
            for depth in depth_idx:
                # Note here we randomly sample time moment 
                # for every combination of factor and depth
                # So, consequetive snapshots are not correlated (on average)
                # Batch is a dataset consisting of one 2D slice of data
                batch = dataset[f'train-{factor}'].select2d(zl=depth)
            
                ############## Training step ###############
                SGSx, SGSy, SGS_norm = get_SGS(batch)

                ######## Optionally, apply symmetries by data augmentation #########
                optimizer.zero_grad()
                MSE_train = MSE(batch, SGSx, SGSy, SGS_norm, ann_Txy, ann_Txx_Tyy,
                                **augment())
                MSE_train.backward()
                optimizer.step()

                del batch

                ############ Validation step ##################
                batch = dataset[f'validate-{factor}'].select2d(zl=depth)
                SGSx, SGSy, SGS_norm = get_SGS(batch)
                with torch.no_grad():
                    MSE_validate = MSE(batch, SGSx, SGSy, SGS_norm, ann_Txy, ann_Txx_Tyy)
                
                del batch
            
                ########### Logging ############
                MSE_train = float(MSE_train.data); MSE_validate = float(MSE_validate.data)
                for key in ['MSE_train', 'MSE_validate']:
                    logger[key][time_iter, ifactor, depth] = eval(key)
                if (time_iter+1) % print_iters == 0:
                    print(f'Factor: {factor}, depth: {depth}, '+'MSE train/validate: [%.6f, %.6f]' % (MSE_train, MSE_validate))
        t = time()
        if (time_iter+1) % print_iters == 0:
            print(f'Iter/num_iters [{time_iter+1}/{time_iters}]. Iter time/Remaining time in seconds: [%.2f/%.1f]' % (t-t_e, (t-t_s)*(time_iters/(time_iter+1)-1)))
        scheduler.step()
    
    return ann_Txy, ann_Txx_Tyy, logger