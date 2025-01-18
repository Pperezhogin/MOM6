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

def get_SGS(batch, predict_smagorinsky=False, Cs_biharm=0.06):
    SGSx = batch.data.SGSx
    SGSy = batch.data.SGSy
    if predict_smagorinsky:
        smag = batch.state.Smagorinsky_biharmonic(Cs_biharm)
        SGSx = SGSx + smag['smagx']
        SGSy = SGSy + smag['smagy']

    SGSx = tensor_from_xarray(SGSx)
    SGSy = tensor_from_xarray(SGSy)

    SGS_norm = 1. / torch.sqrt((SGSx**2 + SGSy**2).mean())
    SGSx = SGSx * SGS_norm
    SGSy = SGSy * SGS_norm

    return SGSx, SGSy, SGS_norm

def MSE(batch, SGSx, SGSy, SGS_norm, ann_Txy, ann_Txx_Tyy, ann_Tall,
        stencil_size=3, dimensional_scaling=True, 
        feature_functions=[], gradient_features=['sh_xy', 'sh_xx', 'vort_xy'],
        rotation=0, reflect_x=False, reflect_y=False,
        short_waves_dissipation=False, short_waves_zero=False,
        batch_perturbed=None,
        response_norm=None, smagx_response=None, smagy_response=None,
        jacobian_trace=False, Cs_biharm=0.06,
        perturbed_inputs=False, jacobian_reduction='component',
        away_from_coast=0):
    prediction = batch.state.Apply_ANN(ann_Txy, ann_Txx_Tyy, ann_Tall,
        stencil_size=stencil_size, dimensional_scaling=dimensional_scaling,
        feature_functions=feature_functions, gradient_features=gradient_features,
        rotation=rotation, reflect_x=reflect_x, reflect_y=reflect_y,
        jacobian_trace=jacobian_trace)

    # If away_from_coast=0, all wet points are included into the loss
    wet_u = propagate_mask(batch.param.wet_u, batch.grid, niter=away_from_coast)
    wet_v = propagate_mask(batch.param.wet_v, batch.grid, niter=away_from_coast)
    areaU = tensor_from_xarray(batch.param.dxCu * batch.param.dyCu * wet_u)
    areaV = tensor_from_xarray(batch.param.dxCv * batch.param.dyCv * wet_v)
    # To make sure that this reduction factor is almost like a mask and of the order of 1
    areaU = areaU / (areaU).max()
    areaV = areaV / (areaV).max()

    def reduction(x,y):
        return (x * areaU + y * areaV).mean()
    
    ANNx = prediction['ZB20u'] * SGS_norm
    ANNy = prediction['ZB20v'] * SGS_norm

    # Go back to classical loss
    #MSE_train = reduction((ANNx-SGSx)**2, (ANNy-SGSy)**2) / reduction((SGSx)**2, (SGSy)**2)
    MSE_train = ((ANNx-SGSx)**2 + (ANNy-SGSy)**2).mean()

    u = tensor_from_xarray(batch.data.u)
    v = tensor_from_xarray(batch.data.v)

    dEdt_error = torch.abs(reduction(u*(ANNx-SGSx), v*(ANNy-SGSy))) / (torch.abs(reduction(u*SGSx,v*SGSy)) + torch.abs(reduction(u*ANNx.detach(),v*ANNy.detach())))

    if short_waves_dissipation:
        perturbed_prediction = batch_perturbed.state.Apply_ANN(ann_Txy, ann_Txx_Tyy, ann_Tall,
                    stencil_size=stencil_size, dimensional_scaling=dimensional_scaling,
                    feature_functions=feature_functions, gradient_features=gradient_features,
                    rotation=rotation, reflect_x=reflect_x, reflect_y=reflect_y)

        ANNx_response = (perturbed_prediction['ZB20u'] - prediction['ZB20u']) * response_norm
        ANNy_response = (perturbed_prediction['ZB20v'] - prediction['ZB20v']) * response_norm
        
        MSE_plane_waves = reduction((ANNx_response - smagx_response)**2, (ANNy_response - smagy_response)**2)
    else:
        MSE_plane_waves = torch.tensor(0)

    if short_waves_zero:
        def fltr(x):
            x = torch_pad(x,right=True, top=True, left=True, bottom=True)
            return (4 * x[1:-1,1:-1] + 2 * (x[2:,1:-1] + x[:-2,1:-1] + x[1:-1,2:] + x[1:-1,:-2]) + (x[2:,2:] + x[2:,:-2] + x[:-2,2:] + x[:-2,:-2])) / 16.
        annx_sharpen = ANNx - fltr(ANNx)
        anny_sharpen = ANNy - fltr(ANNy)
        MSE_short_zero = reduction(annx_sharpen**2, anny_sharpen**2)
    else:
        MSE_short_zero = torch.tensor(0)

    # Note this should not be used with away_from_coast > 0
    if jacobian_trace:
        # First we define mean jacobian trace (per unit grid element)
        # for Smagorinsky model which is analytical
        target_Jtr = - Cs_biharm * 8 * np.sqrt(prediction['sh_xx']**2 + prediction['sh_xy']**2)

        if jacobian_reduction == 'component':
            target_Jtr = target_Jtr.mean()
            MSE_jacobian_trace = \
                (1 - (prediction['dTxx_du'] / target_Jtr).mean())**2 + \
                (1 - (prediction['dTyy_dv'] / target_Jtr).mean())**2 + \
                (1 - (prediction['dTxy_du'] / target_Jtr).mean())**2 + \
                (1 - (prediction['dTxy_dv'] / target_Jtr).mean())**2
        elif jacobian_reduction == 'sum':
            target_Jtr = target_Jtr.mean()
            MSE_jacobian_trace = \
                (1 - 0.25 * (prediction['dTxx_du'] / target_Jtr
                           + prediction['dTyy_dv'] / target_Jtr
                           + prediction['dTxy_du'] / target_Jtr
                           + prediction['dTxy_dv'] / target_Jtr).mean())**2
        elif jacobian_reduction == 'pointwise':
            target_Jtr_norm = 1. / np.sqrt((target_Jtr**2).mean())
            target_Jtr = target_Jtr * target_Jtr_norm

            MSE_jacobian_trace = \
                ((prediction['dTxx_du'] * target_Jtr_norm - target_Jtr)**2 +
                 (prediction['dTyy_dv'] * target_Jtr_norm - target_Jtr)**2 + 
                 (prediction['dTxy_du'] * target_Jtr_norm - target_Jtr)**2 + 
                 (prediction['dTxy_dv'] * target_Jtr_norm - target_Jtr)**2).mean()
        else:
            print('Error: wrong argument')
    else:
        MSE_jacobian_trace = torch.tensor(0)

    if perturbed_inputs:
        perturbed_prediction = batch_perturbed.state.Apply_ANN(ann_Txy, ann_Txx_Tyy, ann_Tall,
                    stencil_size=stencil_size, dimensional_scaling=dimensional_scaling,
                    feature_functions=feature_functions, gradient_features=gradient_features,
                    rotation=rotation, reflect_x=reflect_x, reflect_y=reflect_y)

        ANNx = perturbed_prediction['ZB20u'] * SGS_norm
        ANNy = perturbed_prediction['ZB20v'] * SGS_norm
        MSE_perturbed = reduction((ANNx-SGSx)**2, (ANNy-SGSy)**2)
    else:
        MSE_perturbed = torch.tensor(0)
        
    return MSE_train, MSE_plane_waves, MSE_short_zero, MSE_jacobian_trace, MSE_perturbed, dEdt_error

def train_ANN(factors=[12,15],
              stencil_size = 3,
              hidden_layers=[20],
              dimensional_scaling=True, 
              symmetries='False',
              time_iters=10,
              learning_rate = 1e-3,
              depth_idx=np.arange(1),
              print_iters=1,
              feature_functions=[],
              gradient_features=['sh_xy', 'sh_xx', 'rel_vort'],
              collocated=True,
              permute_factors_and_depth=True,
              short_waves_dissipation=False,
              short_waves_zero=False,
              jacobian_trace=False,
              perturbed_inputs=False,
              grid_harmonic='plane_wave',
              jacobian_reduction='component',
              predict_smagorinsky=False,
              Cs_biharm=0.06,
              away_from_coast=0,
              MSE_weight=1.,
              dEdt_weight=0.,
              load=False,
              subfilter='subfilter-large',
              FGR=3):
    '''
    time_iters is the number of time snaphots
    randomly sampled for each factor and depth

    depth_idx is the indices of the vertical layers which
    participate in training process
    '''
    ########### Read dataset ############
    dataset = read_datasets(['train', 'validate'], factors, load=load, subfilter=subfilter, FGR=FGR)

    ########## Init logger ###########
    logger = xr.Dataset()
    for key in ['MSE_train', 'MSE_plain_waves', 'MSE_short_zero', 'MSE_jacobian_trace', 'MSE_validate', 'MSE_perturbed', 'dEdt_error', 'dEdt_error_validate']:
        logger[key] = xr.DataArray(np.zeros([time_iters, len(factors), len(depth_idx)]), 
                                   dims=['iter', 'factor', 'depth'], 
                                   coords={'factor': factors, 'depth': depth_idx})

    ########## Init ANN ##############
    # As default we have 3 input features on a stencil: D, D_hat and vorticity
    num_input_features = stencil_size**2 * len(gradient_features) + len(feature_functions)
    ann_Tall = None; ann_Txy = None; ann_Txx_Tyy = None
    if collocated:
        ann_Tall = ANN([num_input_features, *hidden_layers, 3])
    else:
        ann_Txy = ANN([num_input_features, *hidden_layers, 1])
        ann_Txx_Tyy = ANN([num_input_features, *hidden_layers, 2])

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
    if collocated:
        all_parameters = ann_Tall.parameters()
    else:
        all_parameters = list(ann_Txy.parameters()) + list(ann_Txx_Tyy.parameters())
    optimizer = optim.Adam(all_parameters, lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
            milestones=[int(time_iters/2), int(time_iters*3/4), int(time_iters*7/8)], gamma=0.1)

    t_s = time()
    for time_iter in range(time_iters):
        t_e = time()

        for factor, depth in iterator(factors, depth_idx):
            # Note here we randomly sample time moment 
            # for every combination of factor and depth
            # So, consequetive snapshots are not correlated (on average)
            # Batch is a dataset consisting of one 2D slice of data
            batch = dataset[f'train-{factor}'].select2d(zl=depth)

            if short_waves_dissipation:
                batch_perturbed = batch.perturb_velocities(grid_harmonic)
                smag = batch.state.Smagorinsky(Cs_biharm=Cs_biharm)
                smag_perturbed = batch_perturbed.state.Smagorinsky(Cs_biharm=Cs_biharm)

                smagx_response = tensor_from_xarray(smag_perturbed['smagx']) - tensor_from_xarray(smag['smagx'])
                smagy_response = tensor_from_xarray(smag_perturbed['smagy']) - tensor_from_xarray(smag['smagy'])
                response_norm = 1. / torch.sqrt((smagx_response**2 + smagy_response**2).mean())
                smagx_response = smagx_response * response_norm
                smagy_response = smagy_response * response_norm
            elif perturbed_inputs:
                batch_perturbed = batch.perturb_velocities('white_noise', amp=0.01)
                response_norm=None; smagx_response=None; smagy_response=None
            else:
                batch_perturbed = None; response_norm=None; smagx_response=None; smagy_response=None

            ############## Training step ###############
            SGSx, SGSy, SGS_norm = get_SGS(batch, predict_smagorinsky=predict_smagorinsky, Cs_biharm=Cs_biharm)

            ######## Optionally, apply symmetries by data augmentation #########
            for rotation, reflect_x, reflect_y in augment():
                optimizer.zero_grad()
                MSE_train, MSE_plain_waves, MSE_short_zero, MSE_jacobian_trace, MSE_perturbed, dEdt_error = \
                            MSE(batch, SGSx, SGSy, SGS_norm, ann_Txy, ann_Txx_Tyy, ann_Tall,
                                stencil_size=stencil_size, dimensional_scaling=dimensional_scaling,
                                feature_functions=feature_functions, gradient_features=gradient_features,
                                rotation=rotation, reflect_x=reflect_x, reflect_y=reflect_y,
                                short_waves_dissipation=short_waves_dissipation, short_waves_zero=short_waves_zero,
                                batch_perturbed=batch_perturbed,
                                response_norm=response_norm, smagx_response=smagx_response, smagy_response=smagy_response,
                                jacobian_trace=jacobian_trace, Cs_biharm=Cs_biharm,
                                perturbed_inputs=perturbed_inputs, jacobian_reduction=jacobian_reduction,
                                away_from_coast=away_from_coast
                                )
                if short_waves_dissipation:
                    (MSE_train + MSE_plain_waves).backward()
                elif short_waves_zero:
                    (MSE_train + MSE_short_zero).backward()
                elif jacobian_trace:
                    (MSE_train + MSE_jacobian_trace).backward()
                elif perturbed_inputs:
                    (MSE_perturbed).backward()
                elif dEdt_weight > 0.:    
                    (MSE_weight * MSE_train + dEdt_weight * dEdt_error).backward()
                else:
                    MSE_train.backward()
                optimizer.step()

            del batch
            del batch_perturbed

            ############ Validation step ##################
            batch = dataset[f'validate-{factor}'].select2d(zl=depth)
            SGSx, SGSy, SGS_norm = get_SGS(batch)
            with torch.no_grad():
                MSE_validate, _, _, _, _, dEdt_error_validate = MSE(batch, SGSx, SGSy, SGS_norm, ann_Txy, ann_Txx_Tyy, ann_Tall,
                                    stencil_size=stencil_size, dimensional_scaling=dimensional_scaling,
                                    feature_functions=feature_functions, gradient_features=gradient_features,
                                    away_from_coast=away_from_coast)
            
            del batch
        
            ########### Logging ############
            MSE_train = float(MSE_train.data)
            MSE_validate = float(MSE_validate.data)
            MSE_perturbed = float(MSE_perturbed.data)
            MSE_plain_waves = float(MSE_plain_waves.data)
            MSE_short_zero = float(MSE_short_zero.data)
            MSE_jacobian_trace = float(MSE_jacobian_trace.data)
            dEdt_error = float(dEdt_error.data)
            dEdt_error_validate = float(dEdt_error_validate.data)

            for key in ['MSE_train', 'MSE_plain_waves', 'MSE_short_zero', 'MSE_jacobian_trace', 'MSE_validate', 'MSE_perturbed', 'dEdt_error', 'dEdt_error_validate']:
                logger[key].loc[{'iter': time_iter, 'factor': factor, 'depth': depth}] = eval(key)
            if (time_iter+1) % print_iters == 0:
                print(f'Factor: {factor}, depth: {depth}, '+'MSE train/validate: [%.6f, %.6f], dEdt error: train/validate [%.6f, %.6f]' % (MSE_train, MSE_validate, dEdt_error, dEdt_error_validate))
        t = time()
        if (time_iter+1) % print_iters == 0:
            print(f'Iter/num_iters [{time_iter+1}/{time_iters}]. Iter time/Remaining time in seconds: [%.2f/%.1f]' % (t-t_e, (t-t_s)*(time_iters/(time_iter+1)-1)))
        scheduler.step()

    for factor in factors:
        for train_str in ['train', 'validate']:
            del dataset[f'{train_str}-{factor}']
    
    return ann_Txy, ann_Txx_Tyy, ann_Tall, logger