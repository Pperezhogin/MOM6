from intake import open_catalog
import xarray as xr
from xgcm import Grid
import numpy as np

def mask_from_nans(variable):
    mask = (1 - np.isnan(variable).astype('float32'))
    if 'time' in variable.dims:
        mask = mask.isel(time=-1)
    if 'time' in variable.coords:
        mask = mask.drop('time')
    return mask


class DatasetCM26():
    def __init__(self):
        rename = {'xt_ocean': 'xh', 'yt_ocean': 'yh', 'xu_ocean': 'xq', 'yu_ocean': 'yq'}
        rename_param = {'dxt': 'dxT', 'dyt': 'dyT', 'dxu': 'dxBu', 'dyu': 'dyBu'}
        
        url = "gs://leap-persistent-ro/groundpepper/GFDL_cm2.6/GFDL_CM2_6_CONTROL_DAILY_SURF.zarr"
        ds = xr.open_dataset(url, engine='zarr', chunks={}, use_cftime=True).rename(**rename).chunk({'yh':-1, 'yq':-1})
        
        cat = open_catalog("https://raw.githubusercontent.com/pangeo-data/pangeo-datastore/master/intake-catalogs/ocean/GFDL_CM2.6.yaml")
        param_init  = cat["GFDL_CM2_6_grid"].to_dask().rename(**rename, **rename_param).chunk({'yh':-1, 'yq':-1})

        param = xr.Dataset()
        for key in ['xh', 'yh', 'xq', 'yq']:
            param[key] = param_init[key]
        for key in ['dxT', 'dyT']:
            param[key] = param_init[key].drop(['area_t', 'dxT', 'dyT', 'geolat_t', 'geolon_t', 'ht', 'kmt', 'wet'])
        for key in ['dxBu', 'dyBu']:
            param[key] = param_init[key].drop(['area_u', 'dxBu', 'dyBu', 'geolat_c', 'geolon_c', 'hu', 'kmu'])
            
        grid = Grid(param, coords={
            'X': {'center': 'xh', 'right': 'xq'},
            'Y': {'center': 'yh', 'right': 'yq'}
            },
           boundary={"X": 'periodic', 'Y': 'fill'},
           fill_value = {'X':1e+20, 'Y':1e+20})
        
        param['wet'] = mask_from_nans(ds.surface_salt)
        param['wet_c'] = mask_from_nans(ds.usurf)
        param['wet_u'] = mask_from_nans(grid.interp(ds.surface_salt,'X')) # We interpolate center variable as it gives best treatment of the coastline
        param['wet_v'] = mask_from_nans(grid.interp(ds.surface_salt,'Y')) # We interpolate center variable as it gives best treatment of the coastline
        
        param['dxCu'] = grid.interp(param.dxT,'X')
        param['dyCu'] = grid.interp(param.dyT,'X')
        param['dxCv'] = grid.interp(param.dxT,'Y')
        param['dyCv'] = grid.interp(param.dyT,'Y')
        
        self.param = param
        
        self.data = xr.Dataset()
        
        self.data['u'] = grid.interp(ds.usurf,'Y').fillna(0.)
        self.data['v'] = grid.interp(ds.vsurf,'X').fillna(0.)
        return