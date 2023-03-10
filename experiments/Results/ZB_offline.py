import xarray as xr
import numpy as np
from helpers.collection_of_experiments import CollectionOfExperiments
from helpers.computational_tools import remesh, select_LatLon, Lk_error, x_coord, y_coord, diffx_tv, diffx_uq, diffy_vq, diffy_tu, filter_apply, filter_iteration, compute_isotropic_KE, compute_isotropic_cospectrum, optimal_amplitude

ds = CollectionOfExperiments.init_folder('/scratch/pp2681/mom6/Feb2022/bare', additional_subfolder='output')

ds.remesh('R64', 'R4')
exp = ds['R64_R4']

exp.u.compute()
exp.v.compute()
exp.h.compute()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ZB_type', type=int, default=0)
parser.add_argument('--ZB_cons', type=int, default=1)
parser.add_argument('--LPF_iter', type=int, default=0)
parser.add_argument('--LPF_order', type=int, default=1)
parser.add_argument('--HPF_iter', type=int, default=0)
parser.add_argument('--HPF_order', type=int, default=1)
parser.add_argument('--Stress_iter', type=int, default=0)
parser.add_argument('--Stress_order', type=int, default=1)
parser.add_argument('--file', type=str, default='ZB_offline.nc')
args = parser.parse_args()

print(args)

ZBx, ZBy = exp.ZB_offline(
            ZB_type=args.ZB_type, ZB_cons=args.ZB_cons, 
            LPF_iter=args.LPF_iter, LPF_order=args.LPF_order,
            HPF_iter=args.HPF_iter, HPF_order=args.HPF_order,
            Stress_iter=args.Stress_iter, Stress_order=args.Stress_order)

Smagx, Smagy = exp.Smagorinsky()

Averaging_Time = slice(3650,7300)
ZB_transfer = compute_isotropic_cospectrum(exp.u, exp.v, ZBx, ZBy,
            exp.param.dxT, exp.param.dyT).sel(Time=Averaging_Time).mean(dim='Time')
ZB_power = 2 * compute_isotropic_KE(ZBx, ZBy, exp.param.dxT, exp.param.dyT).sel(Time=Averaging_Time).mean(dim='Time')

SGS_transfer = exp.SGS_transfer
SGS_power = exp.SGS_power

Smag_transfer = compute_isotropic_cospectrum(exp.u, exp.v, Smagx, Smagy,
            exp.param.dxT, exp.param.dyT).sel(Time=Averaging_Time).mean(dim='Time')
Smag_power = 2 * compute_isotropic_KE(Smagx, Smagy, exp.param.dxT, exp.param.dyT).sel(Time=Averaging_Time).mean(dim='Time')

amp_Eng = (SGS_transfer-Smag_transfer).sum('freq_r') / (ZB_transfer).sum('freq_r')

amp_MSE, MSE, MSE_Eng, corr = optimal_amplitude(ZBx,ZBy,Smagx,Smagy,exp.SGSx,exp.SGSy,exp.u,exp.v,amp_Eng)

dataset = xr.Dataset({
        'ZBx': ZBx, 'ZBy': ZBy,
        'ZB_transfer': ZB_transfer, 'ZB_power': ZB_power, 
        'SGS_transfer': SGS_transfer, 'SGS_power': SGS_power, 
        'Smag_transfer': Smag_transfer, 'Smag_power': Smag_power, 
        'amp_MSE': amp_MSE, 'amp_Eng': amp_Eng, 
        'MSE': MSE, 'MSE_Eng': MSE_Eng, 
        'corr': corr})
dataset.expand_dims(['ZB_type', 'ZB_cons', 'LPF_iter', 'LPF_order', 'HPF_iter', 'HPF_order', 'Stress_iter', 'Stress_order'])
dataset['ZB_type'] = [args.ZB_type]
dataset['ZB_cons'] = [args.ZB_cons]
dataset['LPF_iter'] = [args.LPF_iter]
dataset['LPF_order'] = [args.LPF_order]
dataset['HPF_iter'] = [args.HPF_iter]
dataset['HPF_order'] = [args.HPF_order]
dataset['Stress_iter'] = [args.Stress_iter]
dataset['Stress_order'] = [args.Stress_order]

dataset.to_netcdf(args.file)