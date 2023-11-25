import cartopy.crs as ccrs
import cmocean
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

def x_coord(array):
    '''
    Returns horizontal coordinate, 'xq' or 'xh'
    as xarray
    '''
    try:
        coord = array.xq
    except:
        coord = array.xh
    return coord

def y_coord(array):
    '''
    Returns horizontal coordinate, 'yq' or 'yh'
    as xarray
    '''
    try:
        coord = array.yq
    except:
        coord = array.yh
    return coord

def select_LatLon(array, Lat=(35,45), Lon=(5,15), time=None):
    '''
    array is xarray
    Lat, Lon = tuples of floats
    '''
    x = x_coord(array)
    y = y_coord(array)

    x = array.sel({x.name: slice(Lon[0],Lon[1]), 
                      y.name: slice(Lat[0],Lat[1])})
    
    if 'time' in x.dims:
        if time is None:
            x = x.isel(time=-1)
        else:
            x = x.isel(time=time)
    return x

def select_NA(array, time=None):
    return select_LatLon(array, Lat=(15, 65), Lon=(-90,-10), time=time)

def select_Pacific(array, time=None):
    return select_LatLon(array, Lat=(0, 45), Lon=(-250,-180), time=time)

def select_Cem(array, time=None):
    return select_LatLon(array, Lat=(-10,15), Lon=(-260,-230), time=time)

def select_globe(array, time=None):
    return select_LatLon(array, Lat=(None,None), Lon=(None,None), time=time)

def plot(control, mask=None, vmax=None, selector=select_NA, cartopy=True):
    if mask is not None:
        mask_nan = selector(mask).data.copy()
        mask_nan[mask_nan==0.] = np.nan
    else:
        mask_nan = 1

    control = (mask_nan * selector(control)).compute()
    
    if vmax is None:
        vmax = control.std() * 4
    
    central_latitude = float(y_coord(control).mean())
    central_longitude = float(x_coord(control).mean())
    if cartopy:
        fig, ax = plt.subplots(1,1, figsize=(6, 6), subplot_kw={'projection': ccrs.Orthographic(central_latitude=central_latitude, central_longitude=central_longitude)})
        ax.coastlines(); gl = ax.gridlines(); gl.bottom_labels=True; gl.left_labels=True
        kw = {'transform': ccrs.PlateCarree()}
    else:
        ax = plt.gca()
        kw = {}
    cmap = cmocean.cm.balance
    cmap.set_bad('gray')
    im = selector(control).plot(ax=ax, vmax=vmax, cmap=cmap, add_colorbar=False, **kw)
    plt.tight_layout()
    plt.title('')
    return im

# We compare masked fields because outside there may be 1e+20 values
def compare(tested, control, mask=None, vmax=None, selector=select_NA):
    if mask is not None:
        mask_nan = mask.data.copy()
        mask_nan[mask_nan==0.] = np.nan
        tested = tested * mask_nan
        control = control * mask_nan
    tested = selector(tested).compute()
    control = selector(control).compute()
    
    if vmax is None:
        vmax = control.std() * 4
    
    central_latitude = float(y_coord(control).mean())
    central_longitude = float(x_coord(control).mean())
    fig, axes = plt.subplots(2,2, figsize=(12, 10), subplot_kw={'projection': ccrs.Orthographic(central_latitude=central_latitude, central_longitude=central_longitude)})
    cmap = cmocean.cm.balance
    cmap.set_bad('gray')
    
    ax = axes[0][0]; ax.coastlines(); gl = ax.gridlines(); gl.bottom_labels=True; gl.left_labels=True;
    im = tested.plot(ax=ax, vmax=vmax, transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=False)
    ax.set_title('Tested field')
    ax = axes[0][1]; ax.coastlines(); gl = ax.gridlines(); gl.bottom_labels=True; gl.left_labels=True;
    control.plot(ax=ax, vmax=vmax, transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=False)
    ax.set_title('Control field')
    ax = axes[1][0]; ax.coastlines(); gl = ax.gridlines(); gl.bottom_labels=True; gl.left_labels=True;
    (tested-control).plot(ax=ax, vmax=vmax, transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=False)
    ax.set_title('Tested-control')
    plt.tight_layout()
    plt.colorbar(im, ax=axes, shrink=0.9, aspect=30, extend='both')
    axes[1][1].remove()
    
    ########## Metrics ##############
    error = tested-control
    relative_error = np.abs(error).mean() / np.abs(control).mean()
    R2 = 1 - (error**2).mean() / (control**2).mean()
    optimal_scaling = (tested*control).mean() / (tested**2).mean()
    error = tested * optimal_scaling - control
    R2_max = 1 - (error**2).mean() / (control**2).mean()
    corr = xr.corr(tested, control)
    print('Correlation:', float(corr))
    print('Relative Error:', float(relative_error))
    print('R2 = ', float(R2))
    print('R2 max = ', float(R2_max))
    print('Optinal scaling:', float(optimal_scaling))
    print(f'Nans [test/control]: [{int(np.sum(np.isnan(tested)))}, {int(np.sum(np.isnan(control)))}]')