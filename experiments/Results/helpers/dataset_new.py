import xarray as xr
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import numpy.fft as npfft
import math
import os
from matplotlib import cm

def safe_write(ds, path, safe):
    if (safe):
        exists = os.path.exists(path)
        if exists:
            print('File EXISTS:')
            print(path)
        else:
            print('File does not exist:')
            print(path)   

        print(ds)
        
        input('Press enter...')
        print()
    else:
        print(path)
    ds.to_netcdf(path)

def compute_statistics(folder, tstart=3650., coarsegraining=[], safe=False):
    if isinstance(folder,list):
        for fld in folder:
            compute_statistics(fld, tstart, coarsegraining, safe)
        return

    prog = xr.open_mfdataset(os.path.join(folder,'prog_*.nc'), decode_times=False, parallel=True, concat_dim='Time')
    
    ssh = prog.e.sel(Time=slice(tstart, None))
    data_vars = {'ssh_mean': ssh.mean(dim="Time"), 'ssh_var': ssh.var(dim="Time")}
    ds = xr.Dataset(data_vars=data_vars)
    safe_write(ds, os.path.join(folder,'statistics.nc'), safe)

    for res in coarsegraining:
        Nfactor = int(res[1:])
        grain = int(len(ssh.yh) / (20 * Nfactor))
        if grain < 2:
            continue
        ssh_c = ssh.coarsen(xh=grain, yh=grain).mean()
        data_vars = {'ssh_mean': ssh_c.mean(dim="Time"), 'ssh_var': ssh_c.var(dim="Time")}       
        ds = xr.Dataset(data_vars=data_vars)
        safe_write(ds, os.path.join(folder,'statistics_'+res+'.nc'), safe)

    prog.close()

    return

def plot_error_pcolor(fig, ax, ds, key, zi, C_S, amp, cmap_label, title):
    def get_EXP(x0, y0, x, y):
        idx = (x==x0).nonzero()
        idy = (y==y0).nonzero()
        return np.intersect1d(idx,idy)      

    nexp = 110
    smag = C_S[1:nexp+1]
    amplitude = amp[1:nexp+1]

    x = np.sort(np.unique(smag))
    y = np.sort(np.unique(amplitude))

    X,Y = np.meshgrid(x,y)

    Z = np.zeros_like(X)

    for j in range(X.shape[0]):
        for i in range(X.shape[1]):
            id = 1 + get_EXP(X[j,i],Y[j,i],smag,amplitude)
            Z[j,i] = ds[key].isel(zi=zi, EXP=id)

    im = ax.pcolor(X,Y,Z, cmap=cm.Blues)

    mask = Z==Z.min()
    x0 = float(X[mask])
    y0 = float(Y[mask])
    ax.annotate(f'$C_S={x0:.2f}, \gamma={y0:.5f}$', (x0,y0+0.025))
    ax.plot(x0,y0,'k*',markersize=10)

    ax.set_xlabel('$C_S$')
    ax.set_ylabel(r'$\gamma$, where $\kappa_{BC}=-\gamma dxdy$')
    fig.colorbar(im, ax=ax, label=cmap_label)
    ax.set_title(title)
    return X,Y,Z
