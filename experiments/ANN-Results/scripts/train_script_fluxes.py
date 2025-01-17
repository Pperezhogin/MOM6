import sys
sys.path.append('../')
import numpy as np
import xarray as xr
import torch
from helpers.cm26 import read_datasets
from helpers.train_ann_fluxes import train_ANN_fluxes
from helpers.feature_extractors import *
from helpers.ann_tools import ANN, export_ANN
import json
import gc

import os
import argparse

if __name__ == '__main__':
    ########## Manual input of parameters ###############
    parser = argparse.ArgumentParser()
    parser.add_argument('--stencil_size', type=int, default=3)
    parser.add_argument('--hidden_layers', type=str, default='[20]')
    
    parser.add_argument('--gradient_features', type=str, default="['sh_xy', 'sh_xx', 'rel_vort']")

    parser.add_argument('--subfilter', type=str, default='subfilter-large')
    parser.add_argument('--FGR', type=int, default=3)
    parser.add_argument('--factors', type=str, default='[4,9,12,15]')
    parser.add_argument('--depth_idx', type=str, default='np.arange(10)')
    parser.add_argument('--symmetries', type=str, default='All')
    parser.add_argument('--time_iters', type=int, default=400)
    parser.add_argument('--print_iters', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--permute_factors_and_depth', type=str, default='True')

    parser.add_argument('--path_save', type=str, default='EXP0')

    args = parser.parse_args()

    path_save = f'/scratch/pp2681/mom6/CM26_ML_models/ocean3d/{args.subfilter}/FGR{args.FGR}/{args.path_save}'

    os.system(f'mkdir -p {path_save}/model')

    print(args, '\n')
    with open(f'{path_save}/configuration.txt', "w") as outfile: 
        json.dump(vars(args), outfile)

    args.factors = eval(args.factors)
    args.hidden_layers = eval(args.hidden_layers)
    args.depth_idx = eval(args.depth_idx)
    args.gradient_features = eval(args.gradient_features)
    args.permute_factors_and_depth = eval(args.permute_factors_and_depth)
    
    ann_Tall, logger = \
        train_ANN_fluxes(args.factors,
                  args.stencil_size,
                  args.hidden_layers,
                  args.symmetries,
                  args.time_iters,
                  args.learning_rate,
                  args.depth_idx,
                  args.print_iters,
                  args.gradient_features,
                  args.permute_factors_and_depth,
                  args.subfilter,
                  args.FGR
                  )
    
    nfeatures = ann_Tall.layer_sizes[0]
    export_ANN(ann_Tall, input_norms=torch.ones(nfeatures), output_norms=torch.ones(3), 
            filename=f'{path_save}/model/Tall.nc')
    
    logger.to_netcdf(f'{path_save}/model/logger.nc')

    ds = read_datasets(['test'], [4,9,12,15], subfilter=args.subfilter, FGR=args.FGR)
    os.system(f'mkdir -p {path_save}/skill-test')
    for factor in [4,9,12,15]:
        skill = ds[f'test-{factor}'].predict_ANN(None, None, ann_Tall,
                                                 stencil_size=args.stencil_size,
                                                 gradient_features=args.gradient_features).SGS_skill()
        skill.to_netcdf(f'{path_save}/skill-test/factor-{factor}.nc')
        del skill
        gc.collect()
        print(f'Testing on dataset with factor {factor} is complete')