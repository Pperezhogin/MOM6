import sys
sys.path.append('../')
import numpy as np
import xarray as xr
import torch
from helpers.cm26 import read_datasets
from helpers.train_ann import train_ANN
from helpers.feature_extractors import *
from helpers.ann_tools import ANN, export_ANN
import json

import os
import argparse

if __name__ == '__main__':
    ########## Manual input of parameters ###############
    parser = argparse.ArgumentParser()
    parser.add_argument('--factors', type=str, default='[4,9,12,15]')
    parser.add_argument('--stencil_size', type=int, default=3)
    parser.add_argument('--hidden_layers', type=str, default='[64,64]')
    parser.add_argument('--dimensional_scaling', type=str, default='True')
    parser.add_argument('--symmetries', type=str, default='True')
    parser.add_argument('--time_iters', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--depth_idx', type=str, default='np.arange(10)')
    parser.add_argument('--print_iters', type=int, default=10)
    parser.add_argument('--feature_functions', type=str, default='[]')
    parser.add_argument('--gradient_features', type=str, default="['sh_xy', 'sh_xx', 'vort_xy']")
    parser.add_argument('--collocated', type=str, default='False')
    
    parser.add_argument('--path_save', type=str, default='/scratch/pp2681/mom6/CM26_ML_models/ocean3d/Gauss-FGR3/EXP0')

    args = parser.parse_args()

    os.system(f'mkdir -p {args.path_save}/model')

    print(args, '\n')
    with open(f'{args.path_save}/configuration.txt', "w") as outfile: 
        json.dump(vars(args), outfile)

    args.factors = eval(args.factors)
    args.hidden_layers = eval(args.hidden_layers)
    args.dimensional_scaling = eval(args.dimensional_scaling)
    args.symmetries = eval(args.symmetries)
    args.depth_idx = eval(args.depth_idx)
    args.feature_functions = eval(args.feature_functions)
    args.gradient_features = eval(args.gradient_features)
    args.collocated = eval(args.collocated)

    ann_Txy, ann_Txx_Tyy, ann_Tall, logger = \
        train_ANN(args.factors,
                  args.stencil_size,
                  args.hidden_layers,
                  args.dimensional_scaling, 
                  args.symmetries,
                  args.time_iters,
                  args.learning_rate,
                  args.depth_idx,
                  args.print_iters,
                  args.feature_functions,
                  args.gradient_features,
                  args.collocated)
    
    if args.collocated:
        nfeatures = ann_Tall.layer_sizes[0]
        export_ANN(ann_Tall, input_norms=torch.ones(nfeatures), output_norms=torch.ones(3), 
                filename=f'{args.path_save}/model/Tall.nc')
    else:
        nfeatures = ann_Txy.layer_sizes[0]
        export_ANN(ann_Txy, input_norms=torch.ones(nfeatures), output_norms=torch.ones(1), 
                filename=f'{args.path_save}/model/Txy.nc')
        export_ANN(ann_Txx_Tyy, input_norms=torch.ones(nfeatures), output_norms=torch.ones(2), 
                filename=f'{args.path_save}/model/Txx_Tyy.nc')
    
    logger.to_netcdf(f'{args.path_save}/model/logger.nc')

    ds = read_datasets(['test'], [4,9,12,15])
    os.system(f'mkdir -p {args.path_save}/skill-test')
    for factor in [4,9,12,15]:
        skill = ds[f'test-{factor}'].predict_ANN(ann_Txy, ann_Txx_Tyy, ann_Tall,
                                                 stencil_size=args.stencil_size, dimensional_scaling=args.dimensional_scaling,
                                                 feature_functions=args.feature_functions, gradient_features=args.gradient_features).SGS_skill()
        skill.to_netcdf(f'{args.path_save}/skill-test/factor-{factor}.nc')
        print(f'Testing on dataset with factor {factor} is complete')