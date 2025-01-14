import sys
sys.path.append('../')
import numpy as np
from helpers.cm26 import DatasetCM26
from helpers.operators import *
import os
from time import time
import argparse
import json
import gcm_filters

depth_selector = lambda x: x.isel(zl=np.arange(0,50,5))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--factor', type=int, default=4) # Allowable factors which are close to Cem (4,8,12,16): 4, 9, 12, 15
    parser.add_argument('--FGR', type=int, default=3)
    parser.add_argument('--percentile', type=float, default=0.5)
    parser.add_argument('--coarsening_str', type=str, default='CoarsenKochkovMinMax()')
    parser.add_argument('--subfilter', type=str, default='subfilter')
    parser.add_argument('--filtering_str', type=str, default='Filtering()') # default: Filtering(shape=gcm_filters.FilterShape.TAPER)
    args = parser.parse_args()
    print(args)

    base_path = '/vast/pp2681/CM26_datasets/ocean3d'
    folder = os.path.join(base_path, args.subfilter+'-large', f'FGR{args.FGR}/factor-{args.factor}')
    os.system(f'mkdir -p {folder}')
    with open(f'{folder}/filter.txt', "w") as outfile: 
        json.dump(vars(args), outfile)

    if args.FGR<0:
        args.FGR = None # Will ignore this parameter in subgrid forcing

    for ds_str in ['train', 'validate', 'test']:
        ds = DatasetCM26(source=f'3d-{ds_str}')
        if args.subfilter == 'subfilter':
            SGS_function = ds.compute_subfilter_forcing
        else:
            SGS_function = ds.compute_subgrid_forcing

        coarse_dataset = SGS_function(factor=args.factor, FGR_multiplier=args.FGR, 
                        coarsening=eval(args.coarsening_str), 
                        filtering=eval(args.filtering_str),
                        percentile=args.percentile)
        
        if ds_str == 'train':
            depth_selector(coarse_dataset.param).to_netcdf(os.path.join(folder,'param.nc'))
        
        data, data_constant = coarse_dataset.state.prepare_features()
        data = depth_selector(data)

        if ds_str == 'train':
            depth_selector(data_constant).to_netcdf(os.path.join(folder, 'permanent_features.nc'))

        t_s = time()
        steps = len(data.time)
        for step in range(steps):
            t_e = time()
            data.isel(time=step).to_netcdf(os.path.join(folder, f'{ds_str}-{step}.nc'))
            t = time()
            print(f'{ds_str}: [{step+1}/{steps}]'+', Step time/ETA: [%d/%d]' % (t-t_e, (t-t_s)*(steps/(step+1)-1)))