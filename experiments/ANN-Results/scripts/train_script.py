import sys
sys.path.append('../')
import numpy as np
import xarray as xr
from helpers.cm26 import DatasetCM26, read_datasets
from helpers.ann_tools import ANN, import_ANN, export_ANN, tensor_from_xarray
import torch
import torch.optim as optim
import json
import itertools

import os
from time import time
import argparse

def get_SGS(batch):
    SGSx = tensor_from_xarray(batch.data.SGSx)
    SGSy = tensor_from_xarray(batch.data.SGSy)

    SGS_norm = 1. / torch.sqrt((SGSx**2 + SGSy**2).mean())
    SGSx = SGSx * SGS_norm
    SGSy = SGSy * SGS_norm

    return SGSx, SGSy, SGS_norm

def MSE(batch, SGSx, SGSy, SGS_norm, ann_Txy, ann_Txx_Tyy, rotation=0, reflect_x=False, reflect_y=False, time_revers=False):
    prediction = batch.state.Apply_ANN(ann_Txy, ann_Txx_Tyy,
        rotation=rotation, reflect_x=reflect_x, reflect_y=reflect_y, time_revers=time_revers)

    ANNx = prediction['ZB20u'] * SGS_norm
    ANNy = prediction['ZB20v'] * SGS_norm
    MSE_train = ((ANNx-SGSx)**2 + (ANNy-SGSy)**2).mean()
    return MSE_train

def train_ANN(operator_str, factors, 
              num_epochs, print_epochs, 
              save_epochs, path_save,
              hidden_layers, 
              symmetries, time_revers):
    ########### Read dataset ############
    dataset = read_datasets(operator_str, factors)

    ########## Init ANN ##############
    log_dict = {}
    ann_Txy = ANN([27] + hidden_layers + [1])
    ann_Txx_Tyy = ANN([27] + hidden_layers + [2])
    
    ########## Init iterator for symmetries ######
    if symmetries:
        rots  = [90, 0]
        refxs = [True, False]
        refys = [True, False]
    else:
        rots  = [0]
        refxs = [False]
        refys = [False]

    if time_revers:
        trevs = [True, False]
    else:
        trevs = [False]

    print('Training iterator: ', rots, refxs, refys, trevs, '\n')

    ############ Init optimizer ##############
    all_parameters = list(ann_Txy.parameters()) + list(ann_Txx_Tyy.parameters())
    optimizer = optim.Adam(all_parameters, lr=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
            milestones=[int(num_epochs/2), int(num_epochs*3/4), int(num_epochs*7/8)], gamma=0.1)

    t_s = time()
    for epoch in range(num_epochs):
        t_e = time()
        
        for factor in factors:
            ############## Training step ###############
            batch = dataset[f'train-{factor}'].split()
            SGSx, SGSy, SGS_norm = get_SGS(batch)

            ######## Optionally, apply symmetries by data augmentation #########
            for rotation, reflect_x, reflect_y, time_revers in itertools.product(rots, refxs, refys, trevs):
                optimizer.zero_grad()
                MSE_train = MSE(batch, SGSx, SGSy, SGS_norm, ann_Txy, ann_Txx_Tyy,
                    rotation=rotation, reflect_x=reflect_x, reflect_y=reflect_y, time_revers=time_revers)
                MSE_train.backward()
                optimizer.step()

            del batch

            ############ Testing step ##################
            batch = dataset[f'test-{factor}'].split()
            SGSx, SGSy, SGS_norm = get_SGS(batch)
            with torch.no_grad():
                MSE_test = MSE(batch, SGSx, SGSy, SGS_norm, ann_Txy, ann_Txx_Tyy)
            
            del batch
            
            ########### Logging ############
            log_dict[f'epoch-{epoch+1}-factor-{factor}'] = dict(MSE_train=float(MSE_train.data),MSE_test=float(MSE_test.data))
            if (epoch+1) % print_epochs == 0:
                print(f'Factor: {factor}. '+'MSE train/test: [%.6f, %.6f]' % (float(MSE_train.data), float(MSE_test.data)))
        t = time()
        if (epoch+1) % print_epochs == 0:
            print(f'Epoch/num_epochs [{epoch+1}/{num_epochs}]. Epoch time/Remaining time in seconds: [%.2f/%.1f]' % (t-t_e, (t-t_s)*(num_epochs/(epoch+1)-1)))
        scheduler.step()
        
        ########### Saving model ###########
        if (epoch+1) % save_epochs == 0:
            os.system(f'mkdir -p {path_save}/model')
            export_ANN(ann_Txy, input_norms=torch.ones(27), output_norms=torch.ones(1), 
               filename=f'{path_save}/model/Txy_epoch_{epoch+1}.nc')
            export_ANN(ann_Txx_Tyy, input_norms=torch.ones(27), output_norms=torch.ones(2), 
               filename=f'{path_save}/model/Txx_Tyy_epoch_{epoch+1}.nc')
            with open(f'{path_save}/model/log_dict_epoch_{epoch+1}.json', 'w') as file:
                json.dump(log_dict, file)

    ######### Offline testing ###########
    os.system(f'mkdir -p {path_save}/skill-test')
    os.system(f'mkdir -p {path_save}/skill-train')
    for factor in factors:
        test = dataset[f'test-{factor}']
        test.predict_ANN(ann_Txy, ann_Txx_Tyy).SGS_skill().to_netcdf(f'{path_save}/skill-test/factor-{factor}.nc')
        train = dataset[f'train-{factor}'].split(slice(0,len(test)))
        train.predict_ANN(ann_Txy, ann_Txx_Tyy).SGS_skill().to_netcdf(f'{path_save}/skill-train/factor-{factor}.nc')
        print(f'Offline skill for factor {factor} is saved')

def list_of_experiments(args):
    if args.exp == 'manual':
        pass
    elif args.exp == 'hdn-20':
        pass # It is default experiment

    elif args.exp == 'hdn-64-64':
        args.hidden_layers = [64,64]
        args.path_save = '/scratch/pp2681/mom6/CM26_ML_models/Gauss-FGR2/hdn-64-64'

    elif args.exp == 'hdn-64-64-sym':
        args.hidden_layers = [64,64]
        args.symmetries = True
        args.path_save = '/scratch/pp2681/mom6/CM26_ML_models/Gauss-FGR2/hdn-64-64-sym'

    elif args.exp == 'hdn-64-64-sym-trev':
        args.hidden_layers = [64,64]
        args.symmetries = True
        args.time_revers = True
        args.path_save = '/scratch/pp2681/mom6/CM26_ML_models/Gauss-FGR2/hdn-64-64-sym-trev'
        
    elif args.exp == 'test-1':
        args.factors = [6]
        args.num_epochs = 50
        args.print_epochs = 1
        args.save_epochs = 50
        args.path_save = 'test-1'

    elif args.exp == 'test-2':
        args.factors = [6]
        args.num_epochs = 50
        args.print_epochs = 1
        args.save_epochs = 50
        args.symmetries = True
        args.path_save = 'test-2'

    elif args.exp == 'test-3':
        args.factors = [6]
        args.num_epochs = 50
        args.print_epochs = 1
        args.save_epochs = 50
        args.symmetries = True
        args.time_revers = True
        args.path_save = 'test-3'

    elif args.exp == 'test-4':
        args.factors = [12]
        args.num_epochs = 1
        args.print_epochs = 1
        args.save_epochs = 1
        args.path_save = 'test-4'

    else:
        raise ValueError(f'Experiment {args.exp} is not defined')
    
    return args

if __name__ == '__main__':
    ########## Manual input of parameters ###############
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='manual')
    parser.add_argument('--operator_str', type=str, default='Gauss-FGR2')
    parser.add_argument('--factors', type=str, default='[4,6,9,12]')
    parser.add_argument('--num_epochs', type=int, default=2000)
    parser.add_argument('--print_epochs', type=int, default=10)
    parser.add_argument('--save_epochs', type=int, default=500)
    parser.add_argument('--path_save', type=str, default='/scratch/pp2681/mom6/CM26_ML_models/Gauss-FGR2/hdn-20')
    parser.add_argument('--hidden_layers', type=str, default='[20]')
    parser.add_argument('--symmetries', type=str, default='False')
    parser.add_argument('--time_revers', type=str, default='False')
    args = parser.parse_args()
    args.factors = eval(args.factors)
    args.hidden_layers = eval(args.hidden_layers)
    args.symmetries = eval(args.symmetries)
    args.time_revers = eval(args.time_revers)

    # Check if there is predefined experiment
    args = list_of_experiments(args)

    print(args, '\n')

    train_ANN(args.operator_str, args.factors, 
              args.num_epochs, args.print_epochs, 
              args.save_epochs, args.path_save,
              args.hidden_layers, 
              args.symmetries, args.time_revers)