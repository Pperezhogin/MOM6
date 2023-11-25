import sys
sys.path.append('../')
import numpy as np
import xarray as xr
from helpers.cm26 import DatasetCM26
from helpers.operators import Coarsen, Filtering, Subsampling, CoarsenKochkov, CoarsenWeighted
import os
from time import time
import argparse

def generate_dataset(dataset, dataset_str, factors, operator_str,
    base_path, batch_size):
    steps = int(np.ceil(len(dataset.data.time) / batch_size))
    
    t_s = time()
    for step in range(steps):
        t_e = time()
        idx = slice(step*batch_size,(step+1)*batch_size)
        batch = dataset.sample_batch(time=idx, operator=eval(operator_str), factors=factors)
        for factor in factors:
            folder = os.path.join(base_path, operator_str, f'factor-{factor}')
            path = os.path.join(folder,f'{dataset_str}-{step}.nc')
            os.system(f'mkdir -p \'{folder}\'')
            batch[factor].data.astype('float32').to_netcdf(path)
            
            if step==0 and dataset_str == 'train':
                batch[factor].param.astype('float32').to_netcdf(os.path.join(folder,'param.nc'))
        t = time()
        print(f'{operator_str}-{dataset_str}, steps: [{step+1}/{steps}]'+', Step time/ETA: [%d/%d]' % (t-t_e, (t-t_s)*(steps/(step+1)-1)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--operator_str', type=str, default='CoarsenWeighted()')
    args = parser.parse_args()
    print(args)

    factors = [4,6,9,12]
    base_path = '/scratch/pp2681/mom6/CM26_datasets'
    batch_size = 14

    ds = DatasetCM26()
    ds_train = ds.split(time=slice(0,6209,5))
    ds_test = ds.split(time=slice(6209,7305,5))

    folder = os.path.join(base_path, args.operator_str)
    os.system(f'rm -rf \'{folder}\'')

    generate_dataset(ds_train, 'train', factors, args.operator_str, base_path, batch_size)
    generate_dataset(ds_test, 'test', factors, args.operator_str, base_path, batch_size)