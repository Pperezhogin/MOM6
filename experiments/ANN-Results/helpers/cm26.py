from intake import open_catalog
import xarray as xr
from xgcm import Grid
import numpy as np
from helpers.computational_tools import StateFunctions

def mask_from_nans(variable):
    mask = (1 - np.isnan(variable).astype('float32'))
    if 'time' in variable.dims:
        mask = mask.isel(time=-1)
    if 'time' in variable.coords:
        mask = mask.drop('time')
    return mask

# If there is any point in averaging box, it is treated as land point
def discard_land(x):
    return (x==1).astype('float32')

def remesh(x, hires_ds, ds_coarse):
    if 'xq' in x.dims and 'yh' in x.dims:
        variable = 'U-point'
    if 'xh' in x.dims and 'yq' in x.dims:
        variable = 'V-point'
    
    if variable == 'U-point':
        x_coarse = ds_coarse.grid.interp(hires_ds.grid.interp(x, 'X').coarsen({'xh':ds_coarse.factor, 'yh':ds_coarse.factor}).mean(),'X') * ds_coarse.param.wet_u
        
    if variable == 'V-point':
        x_coarse = ds_coarse.grid.interp(hires_ds.grid.interp(x, 'Y').coarsen({'xh':ds_coarse.factor, 'yh':ds_coarse.factor}).mean(),'Y') * ds_coarse.param.wet_v
    
    return x_coarse
        
        
class DatasetCM26():
    def from_cloud(self):
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
           fill_value = {'X':0, 'Y':0})
        
        param['wet'] = mask_from_nans(ds.surface_salt)
        param['wet_u'] = discard_land(grid.interp(param['wet'], 'X'))
        param['wet_v'] = discard_land(grid.interp(param['wet'], 'Y'))
        param['wet_c'] = discard_land(grid.interp(param['wet'], ['X', 'Y']))
        
        param['dxCu'] = grid.interp(param.dxT,'X')
        param['dyCu'] = grid.interp(param.dyT,'X')
        param['dxCv'] = grid.interp(param.dxT,'Y')
        param['dyCv'] = grid.interp(param.dyT,'Y')
        
        self.param = param
        
        self.data = xr.Dataset()
        
        self.data['u'] = grid.interp(ds.usurf,'Y').fillna(0.) * param.wet_u
        self.data['v'] = grid.interp(ds.vsurf,'X').fillna(0.) * param.wet_v
        
        self.data['time'] = ds['time']
        
        self.grid = grid
        
    def __init__(self, data=None, param=None, grid=None):
        if data is None and param is None:
            self.from_cloud()
        else:
            self.data = data
            self.param = param
            self.grid = grid
        self.state = StateFunctions(self.data, self.param, self.grid)
        return

    def coarsen(self, factor=10):
        ################# Start with coarsening grid steps #####################
        dxT = self.param.dxT.coarsen({'xh':factor}).sum().coarsen({'yh':factor}).mean()
        dyT = self.param.dyT.coarsen({'yh':factor}).sum().coarsen({'xh':factor}).mean()
        
        ############## Manually create C-Arakawa coordinates ###################
        xh = dxT.xh
        xq = xh.data + (xh[1].data - xh[0].data)*0.5
        xq = xr.DataArray(xq, dims=['xq'])
        
        yh = dxT.yh
        yyq = (yh[1:].data + yh[:-1].data)*0.5
        yq = np.zeros(yh.size)
        yq[:-1] = yyq
        yq[-1] = yh[-1].data + (yh[-1].data-yh[-2].data)/2
        
        yq = xr.DataArray(yq, dims=['yq'])
        
        param = xr.Dataset()
        param['xh'] = xh
        param['xq'] = xq
        param['yh'] = yh
        param['yq'] = yq
        
        ########### Initialize Grid and compute grid steps #####################
        
        grid = Grid(param, coords={
            'X': {'center': 'xh', 'right': 'xq'},
            'Y': {'center': 'yh', 'right': 'yq'}
            },
           boundary={"X": 'periodic', 'Y': 'fill'},
           fill_value = {'X':0, 'Y':0})

        param['dxT'] = dxT
        param['dyT'] = dyT
        param['dxCu'] = grid.interp(param.dxT,'X')
        param['dyCu'] = grid.interp(param.dyT,'X')
        param['dxCv'] = grid.interp(param.dxT,'Y')
        param['dyCv'] = grid.interp(param.dyT,'Y')
        param['dxBu'] = grid.interp(param.dxT,['X','Y'])
        param['dyBu'] = grid.interp(param.dyT,['X','Y'])
        
        ######################### Creating wet masks ###########################
        param['wet'] = discard_land(self.param.wet.coarsen({'xh':factor,'yh':factor}).mean())
        param['wet_u'] = discard_land(grid.interp(param['wet'], 'X'))
        param['wet_v'] = discard_land(grid.interp(param['wet'], 'Y'))
        param['wet_c'] = discard_land(grid.interp(param['wet'], ['X', 'Y']))
        
        ##################### Coarsegraining velocities ########################
        
        data = xr.Dataset()

        data['u'] = grid.interp(self.grid.interp(self.data.u, 'X').coarsen({'xh':factor, 'yh':factor}).mean(),'X') * param.wet_u
        data['v'] = grid.interp(self.grid.interp(self.data.v, 'Y').coarsen({'xh':factor, 'yh':factor}).mean(),'Y') * param.wet_v
        
        ds_coarse = DatasetCM26(data, param, grid)
        ds_coarse.factor = factor
  
        return ds_coarse

    def subgrid_forcing(self, ds_coarse):
        hires_advection = self.state.advection()
        advx = remesh(hires_advection[0], self, ds_coarse)
        advy = remesh(hires_advection[1], self, ds_coarse)
        
        coarse_advection = ds_coarse.state.advection()
        SGSx = advx - coarse_advection[0]
        SGSy = advy - coarse_advection[1]
        return {'SGSx': SGSx, 'SGSy': SGSy}
    
    def sample_epoch(self, time=None, cftime=None, factors = [2,4,6,10,20]):
        '''
        This function takes one time snapshot and produces training dataset
        consisting of velocity gradients on a coarse grid and 
        corresponding subgrid forcing, for a range of factors
        '''
        if time is not None and cftime is not None:
            print('Error: use only one selector of time')
            return
        elif time is not None:
            data = self.data.isel(time=time)
        elif cftime is not None:
            data = self.data.sel(time=cftime)
        else:
            print('Time argument is not provided')
            return
        
        # Load a single snapshot for fast processing
        snapshot = DatasetCM26(data.compute(), self.param.compute(), self.grid)
        
        hires_advection = snapshot.state.advection()
        advx = hires_advection[0].compute()
        advy = hires_advection[1].compute()
        
        output = {}
        for factor in factors:
            ds_coarse = snapshot.coarsen(factor)
            coarse_advection = ds_coarse.state.advection()
            
            ds_coarse.data['SGSx'] = remesh(advx, snapshot, ds_coarse) - coarse_advection[0]
            ds_coarse.data['SGSy'] = remesh(advy, snapshot, ds_coarse) - coarse_advection[1]                        
            output[factor] = ds_coarse
                            
        return output        