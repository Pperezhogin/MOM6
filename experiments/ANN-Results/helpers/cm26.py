import xarray as xr
from xgcm import Grid
import numpy as np
from helpers.state_functions import StateFunctions
from helpers.operators import Coarsen, CoarsenWeighted, CoarsenKochkov, Subsampling, Filtering
from functools import cache

def mask_from_nans(variable):
    mask = (1 - np.isnan(variable).astype('float32'))
    if 'time' in variable.dims:
        mask = mask.isel(time=-1)
    if 'time' in variable.coords:
        mask = mask.drop('time')
    return mask

def discard_land(x, percentile=1):
    '''
    Input is the mask array. Supposed that it was
    obtained with interpolation or coarsegraining
    
    percentile controls how to treat land:
    * percentile=1 means that if in an averaging
    box during coarsening there was any land point,
    we treat coarse point as land point
    * percentile=0 means that of in an averaging box
    there was at least one computational point, we 
    treat coarse point as wet point
    * percentile=0.5 means that if in an averaging
    box there were more than half wet points,
    we treat coarse point as wet point
    '''
    if percentile<0 or percentile>1:
        print('Error: choose percentile between 0 and 1')
    if percentile==1:
        return (x==1).astype('float32')
    else:
        return (x>percentile).astype('float32')
    
class DatasetCM26():
    def from_cloud(self, source='cmip6'):
        '''
        Algorithm:
        * Initialize data and grid information from cloud in a lazy way
        * Create C-Awakawa grid
        * Interpolate data to C-Arakawa grid
        * Hint: put wall to the north pole for simplicity
        '''
        ############ Read datasets ###########
        if source == 'leap':
            from intake import open_catalog
            ds = xr.open_dataset("gs://leap-persistent-ro/groundpepper/GFDL_cm2.6/GFDL_CM2_6_CONTROL_DAILY_SURF.zarr", engine='zarr', chunks={}, use_cftime=True)
            cat = open_catalog("https://raw.githubusercontent.com/pangeo-data/pangeo-datastore/master/intake-catalogs/ocean/GFDL_CM2.6.yaml")
            param_init  = cat["GFDL_CM2_6_grid"].to_dask()
        elif source == 'cmip6':
            ds = xr.open_dataset("gs://cmip6/GFDL_CM2_6/control/surface", engine='zarr', chunks={}, use_cftime=True)
            param_init = xr.open_dataset('gs://cmip6/GFDL_CM2_6/grid', engine='zarr')
        else:
            print('Error: wrong source parameter')
        
        ############ Rename coordinates ###########
        rename = {'xt_ocean': 'xh', 'yt_ocean': 'yh', 'xu_ocean': 'xq', 'yu_ocean': 'yq'}
        rename_param = {'dxt': 'dxT', 'dyt': 'dyT', 'dxu': 'dxBu', 'dyu': 'dyBu'}

        ds = ds.rename(**rename).chunk({'yh':-1, 'yq':-1})
        param_init = param_init.rename(**rename, **rename_param).chunk({'yh':-1, 'yq':-1})

        ############ Drop unnecessary coordinates ###########
        param = xr.Dataset()
        for key in ['xh', 'yh', 'xq', 'yq']:
            param[key] = param_init[key]
        for key in ['dxT', 'dyT']:
            param[key] = param_init[key].drop(['area_t', 'dxT', 'dyT', 'geolat_t', 'geolon_t', 'ht', 'kmt', 'wet'])
        for key in ['dxBu', 'dyBu']:
            param[key] = param_init[key].drop(['area_u', 'dxBu', 'dyBu', 'geolat_c', 'geolon_c', 'hu', 'kmu'])
        
        ############ Init xgcm.Grid object for C-grid ###########
        # Note, we implement B.C. only in zonal diretion,
        # but simply set zero B.C. in meridional direction 
        grid = Grid(param, coords={
            'X': {'center': 'xh', 'right': 'xq'},
            'Y': {'center': 'yh', 'right': 'yq'}
            },
           boundary={"X": 'periodic', 'Y': 'fill'},
           fill_value = {'Y':0})
        self.grid = grid
        
        ############ Compute masks for C-grid ###########
        # Note, we assume that coastline goes accross U,V and corner points,
        # while land points are uniquely defined by the T points
        param['wet'] = mask_from_nans(ds.surface_salt)

        # Set manually wall for the layer of points
        # close to the north pole
        param['wet'][{'yh':-1}] = 0

        # Interpolating mask from cell center to corners or sides
        # will result to values less than 1, and thus we mark
        # these points as land
        param['wet_u'] = discard_land(grid.interp(param['wet'], 'X'))
        param['wet_v'] = discard_land(grid.interp(param['wet'], 'Y'))
        param['wet_c'] = discard_land(grid.interp(param['wet'], ['X', 'Y']))
        
        ########### Compute grid steps for C-grid #########
        # Grid steps are computed such that total 
        # length of the grid line is preserved
        # To achieve this, we need to interpolate only along the 
        # grid step "direction"
        # Infer grid information from cell center
        param['dxCu'] = grid.interp(param.dxT,'X')
        param['dyCv'] = grid.interp(param.dyT,'Y')
        
        # Infer grid information from cell corner
        param['dyCu'] = grid.interp(param.dyBu,'Y')
        param['dxCv'] = grid.interp(param.dxBu,'X')
        
        self.param = param.compute()
        
        ########### Interpolate velocities to C-grid ############
        self.data = xr.Dataset()
        self.data['u'] = grid.interp(ds.usurf.fillna(0.) * param.wet_c,'Y') * param.wet_u
        self.data['v'] = grid.interp(ds.vsurf.fillna(0.) * param.wet_c,'X') * param.wet_v
        self.data['time'] = ds['time']
        
    def __init__(self, data=None, param=None, grid=None, source='cmip6'):
        if data is None and param is None:
            self.from_cloud(source=source)
        else:
            self.data = data
            self.param = param
            self.grid = grid
        self.state = StateFunctions(self.data, self.param, self.grid)
        return
    
    def __del__(self):
        del self.data, self.param, self.grid, self.state
        return
    
    def init_coarse_grid(self, factor=10, percentile=0):
        '''
        Here "self" is the DatasetCM26 object
        We cache the coarse grid initialization because it takes
        1 sec to run time function, but the result will be used many times

        Algorithm of coarse grid initialization:
        * Coarsegrain 1D coordinate lines
        * Sum grid steps along grid lines
        * Create xgcm.Grid object
        * Create wet masks
        '''

        ############# Start with coarsening the 1D coordinate lines of the Arawaka-C grid #################
        xh = self.param.xh.coarsen({'xh':factor}).mean()
        yh = self.param.yh.coarsen({'yh':factor}).mean()
        xq = self.param.xq.isel(xq = slice(factor-1,None,factor))
        yq = self.param.yq.isel(yq = slice(factor-1,None,factor))

        ################# Summing grid steps along grid lines #####################
        param = xr.Dataset()
        param['xh'] = xh
        param['xq'] = xq
        param['yh'] = yh
        param['yq'] = yq
        # These four summations are well defined without nans
        param['dxT']  = self.param.dxT.coarsen({'xh':factor}).sum().interp(yh=yh)
        param['dyT']  = self.param.dyT.coarsen({'yh':factor}).sum().interp(xh=xh)
        param['dyCu'] = self.param.dyCu.coarsen({'yh':factor}).sum().interp(xq=xq)
        param['dxCv'] = self.param.dxCv.coarsen({'xh':factor}).sum().interp(yq=yq)
        # These summations require special treatment of B.C.s.
        param['dyBu'] = self.param.dyBu.coarsen({'yq':factor}).sum().interp(xq=xq,yq=yq)
        param['dxBu'] = self.param.dxBu.coarsen({'xq':factor}).sum().interp(xq=xq,yq=yq)
        param['dxBu'][{'xq':-1}] = param['dxBu'][{'xq':-2}] # Because interpolation on the right boundary is not defined
        param['dyBu'][{'yq':-1}] = param['dyBu'][{'yq':-2}] # Because interpolation on the right boundary is not defined

        param['dxCu'] = self.param.dxCu.coarsen({'xq':factor}).sum().interp(xq=xq,yh=yh)
        param['dxCu'][{'xq':-1}] = param['dxCu'][{'xq':-2}] # Because interpolation on the right boundary is not defined
        param['dyCv'] = self.param.dyCv.coarsen({'yq':factor}).sum().interp(yq=yq,xh=xh)
        param['dyCv'][{'yq':-1}] = param['dyCv'][{'yq':-2}] # Because interpolation on the right boundary is not defined
        
        ############ Creating xgcm.Grid object ############
        grid = Grid(param, coords={
            'X': {'center': 'xh', 'right': 'xq'},
            'Y': {'center': 'yh', 'right': 'yq'}
            },
            boundary={"X": 'periodic', 'Y': 'fill'},
            fill_value = {'Y':0})
        
        ######################### Creating wet masks ###########################
        param['wet'] = discard_land(self.param.wet.coarsen({'xh':factor,'yh':factor}).mean(), percentile=percentile)
        # Set manually wall for the layer of points
        # close to the north pole
        param['wet'][{'yh':-1}] = 0
        param['wet_u'] = discard_land(grid.interp(param['wet'], 'X'))
        param['wet_v'] = discard_land(grid.interp(param['wet'], 'Y'))
        param['wet_c'] = discard_land(grid.interp(param['wet'], ['X', 'Y']))

        return param.compute(), grid

    def coarsen(self, factor=10, operator=CoarsenWeighted(), percentile=0, param=None, grid=None):
        '''
        Coarsening of the dataset with a given factor

        Algorithm:
        * Initialize coarse grid
        * Coarsegrain velocities by applying operator
        * Return new dataset with coarse velocities
        '''
        # Initialize coarse grid
        if param is None or grid is None:
            param, grid = self.init_coarse_grid(factor=factor, percentile=percentile)

        ##################### Coarsegraining velocities ########################
        data = xr.Dataset()
        # Create coarse version of the dataset
        ds_coarse = DatasetCM26(data, param, grid)
        ds_coarse.factor = factor
        
        # Coarsegrain velocities
        ds_coarse.data['u'], ds_coarse.data['v'] = operator(self.data.u, self.data.v, self, ds_coarse)
  
        return ds_coarse
  
    def sample_batch(self, time=np.random.randint(0,7305,1), factors = [2,4,6,10,20], operator=CoarsenWeighted(), percentile=0):
        '''
        This function samples batch and produces training dataset
        consisting of velocity gradients on a coarse grid and 
        corresponding subgrid forcing, for a range of factors
        '''
        ############## Automatic generation of coarse grids ###################
        try:
            for factor in factors:
                self.params[factor]; self.grids[factor]
        except:
            self.params = {}; self.grids = {}
            for factor in factors:
                self.params[factor], self.grids[factor] = self.init_coarse_grid(factor=factor, percentile=percentile)

        ############# Sampling batch from the dataset ###################
        data = self.data.isel(time=time)
        batch = DatasetCM26(data.compute(), self.param, self.grid)

        ############# High-resolution advection #################
        hires_advection = batch.state.advection()
        advx = hires_advection[0].compute()
        advy = hires_advection[1].compute()
        
        ############## Coarsegraining and SGS ###################
        output = {}
        for factor in factors:
            ds_coarse = batch.coarsen(factor, operator=operator, percentile=percentile, 
                                      param=self.params[factor], grid=self.grids[factor])
            coarse_advection = ds_coarse.state.advection()
            
            ds_coarse.data['SGSx'], ds_coarse.data['SGSy'] = operator(advx, advy, batch, ds_coarse)
            ds_coarse.data['SGSx'] = ds_coarse.data['SGSx'] - coarse_advection[0]
            ds_coarse.data['SGSy'] = ds_coarse.data['SGSy'] - coarse_advection[1]
            ds_coarse.data = ds_coarse.data.squeeze()
            
            output[factor] = ds_coarse

        return output