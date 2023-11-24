from intake import open_catalog
import xarray as xr
from xgcm import Grid
import numpy as np
from helpers.computational_tools import StateFunctions
import gcm_filters
from xgcm.padding import pad

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

def operator_coarsen(u, v, ds_hires, ds_coarse):
    '''
    Algorithm: 
    * Interpolate velocities to the center
    * Coarsegrain
    * Interpolate to nodes of Arakawa-C grid on coarse grid
    '''
    u_coarse = ds_coarse.grid.interp(
              (ds_hires.grid.interp(u, 'X')*ds_hires.param.wet).coarsen(
              {'xh':ds_coarse.factor, 'yh':ds_coarse.factor}).mean() * ds_coarse.param.wet,'X') * ds_coarse.param.wet_u
    
    v_coarse = ds_coarse.grid.interp(
              (ds_hires.grid.interp(v, 'Y')*ds_hires.param.wet).coarsen(
              {'xh':ds_coarse.factor, 'yh':ds_coarse.factor}).mean() * ds_coarse.param.wet,'Y') * ds_coarse.param.wet_v   
    return u_coarse, v_coarse

def operator_coarsen_weighted(u, v, ds_hires, ds_coarse):
    '''
    Algorithm: 
    * Interpolate velocities to the center
    * Coarsegrain
    * Interpolate to nodes of Arakawa-C grid on coarse grid
    
    But now: all interpolations and coarsenings are weighted with mesh area
    '''
    
    ############ U-velocity ############
    areaU = ds_hires.param.dxCu * ds_hires.param.dyCu
    # u velocity in T point weighted with area
    u_T_weighted = ds_hires.grid.interp(u * areaU,'X') * ds_hires.param.wet
    # Note, multiplication by mask is not needed because:
    # All T points coming from boundary will be discarded on coarse grid
    
    areaU = ds_coarse.param.dxCu * ds_coarse.param.dyCu
    u_coarse = ds_coarse.grid.interp(
        u_T_weighted.coarsen({'xh':ds_coarse.factor, 'yh':ds_coarse.factor}).sum() * ds_coarse.param.wet
        ,'X') \
        * ds_coarse.param.wet_u / areaU

    ############ V-velocity ############
    areaV = ds_hires.param.dxCv * ds_hires.param.dyCv
    # v velocity in T point weighted with area
    v_T_weighted = ds_hires.grid.interp(v * areaV,'Y') * ds_hires.param.wet
    
    areaV = ds_coarse.param.dxCv * ds_coarse.param.dyCv
    v_coarse = ds_coarse.grid.interp(
        v_T_weighted.coarsen({'xh':ds_coarse.factor, 'yh':ds_coarse.factor}).sum() * ds_coarse.param.wet
        ,'Y') \
        * ds_coarse.param.wet_v / areaV
                
    return u_coarse, v_coarse

def operator_gaussian(u, v, ds_hires, ds_coarse, FGR=2):
    '''
    Algorithm: apply top-hat filter with prescribed FGR
    w.r.t. coarse grid step. The top-hat filter is 
    informed with mask and local grid area
    
    The filtered field is downscaled to coarse grid
    with subsampling operation (in a case of missing points,
    with interpolation)
    '''
    FGR_coarse = FGR * ds_coarse.factor
    ############ U-velocity ############
    areaU = ds_hires.param.dxCu * ds_hires.param.dyCu
    filter_simple_fixed_factor = gcm_filters.Filter(
        filter_scale=FGR_coarse,
        dx_min=1,
        filter_shape=gcm_filters.FilterShape.GAUSSIAN,
        grid_type=gcm_filters.GridType.REGULAR_WITH_LAND_AREA_WEIGHTED,
        grid_vars={'area': areaU, 'wet_mask': ds_hires.param.wet_u}
        )
    u_filtered = filter_simple_fixed_factor.apply(u, dims=['yh', 'xq'])
    u_coarse = u_filtered.interp(xq=ds_coarse.param.xq, yh=ds_coarse.param.yh)
    
    ############ V-velocity ############
    areaV = ds_hires.param.dxCv * ds_hires.param.dyCv
    filter_simple_fixed_factor = gcm_filters.Filter(
        filter_scale=FGR_coarse,
        dx_min=1,
        filter_shape=gcm_filters.FilterShape.GAUSSIAN,
        grid_type=gcm_filters.GridType.REGULAR_WITH_LAND_AREA_WEIGHTED,
        grid_vars={'area': areaV, 'wet_mask': ds_hires.param.wet_v}
        )
    v_filtered = filter_simple_fixed_factor.apply(v, dims=['yq', 'xh'])
    v_coarse = v_filtered.interp(xh=ds_coarse.param.xh, yq=ds_coarse.param.yq)
    
    return u_coarse, v_coarse

def operator_Kochkov(u, v, ds_hires, ds_coarse):
    '''
    Algorithm: apply coarsegraining of velocities
    in finite-volume fashion following Kochkov2021 (see their Supplementary). 
    It should help preserve as much information about short
    waves as possible.
    
    Also, it satisfies exactly the incompressibility condition for 
    coarsegrained field
    '''
    factor = ds_coarse.factor
    param = ds_coarse.param
    
    # Here we apply coarsening along y direction,
    # while in x direction we do simple subsampling.
    # Also, we weight with grid steps to satisy
    # incompressibility condition.
    u_coarse = (u*ds_hires.param.dyCu).rolling(yh=factor).sum().isel(
        yh=slice(factor-1,None,factor), xq=slice(factor-1,None,factor))
    # We update coordinates because of strange rolling algorithm and
    # not exact (up to numerics) coincidence for xq coordinate
    u_coarse['yh'] = param.yh
    u_coarse['xq'] = param.xq
    u_coarse = u_coarse / ds_coarse.param.dyCu * ds_coarse.param.wet_u

    # V velocity
    v_coarse = (v*ds_hires.param.dxCv).rolling(xh=factor).sum().isel(
    yq=slice(factor-1,None,factor), xh=slice(factor-1,None,factor))
    v_coarse['yq'] = param.yq
    v_coarse['xh'] = param.xh
    v_coarse = v_coarse / ds_coarse.param.dxCv * ds_coarse.param.wet_v

    return u_coarse, v_coarse
    
class DatasetCM26():
    def from_cloud(self, source='leap'):
        rename = {'xt_ocean': 'xh', 'yt_ocean': 'yh', 'xu_ocean': 'xq', 'yu_ocean': 'yq'}
        rename_param = {'dxt': 'dxT', 'dyt': 'dyT', 'dxu': 'dxBu', 'dyu': 'dyBu'}
        
        if source == 'leap':
            ds = xr.open_dataset("gs://leap-persistent-ro/groundpepper/GFDL_cm2.6/GFDL_CM2_6_CONTROL_DAILY_SURF.zarr", engine='zarr', chunks={}, use_cftime=True)
            cat = open_catalog("https://raw.githubusercontent.com/pangeo-data/pangeo-datastore/master/intake-catalogs/ocean/GFDL_CM2.6.yaml")
            param_init  = cat["GFDL_CM2_6_grid"].to_dask()
        elif source == 'cmip6':
            ds = xr.open_dataset("gs://cmip6/GFDL_CM2_6/control/surface", engine='zarr', chunks={}, use_cftime=True)
            param_init = xr.open_dataset('gs://cmip6/GFDL_CM2_6/grid', engine='zarr')
        else:
            print('Error: wrong source parameter')
            
        ds = ds.rename(**rename).chunk({'yh':-1, 'yq':-1})
        param_init = param_init.rename(**rename, **rename_param).chunk({'yh':-1, 'yq':-1})

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
        
        self.param = param.compute()
        
        self.data = xr.Dataset()
        
        self.data['u'] = grid.interp(ds.usurf,'Y').fillna(0.) * param.wet_u
        self.data['v'] = grid.interp(ds.vsurf,'X').fillna(0.) * param.wet_v
        
        self.data['time'] = ds['time']
        
        self.grid = grid
        
    def __init__(self, data=None, param=None, grid=None, source='leap'):
        if data is None and param is None:
            self.from_cloud(source=source)
        else:
            self.data = data
            self.param = param
            self.grid = grid
        self.state = StateFunctions(self.data, self.param, self.grid)
        return

    def coarsen(self, factor=10, operator=operator_coarsen, percentile=1):
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
        #yq[-1] = yh[-1].data + (yh[-1].data-yh[-2].data)/2
        yq[-1] = 90.0 # This is the property for all CM2.6 grids
        
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
        param['wet'] = discard_land(self.param.wet.coarsen({'xh':factor,'yh':factor}).mean(), percentile=percentile)
        param['wet_u'] = discard_land(grid.interp(param['wet'], 'X'))
        param['wet_v'] = discard_land(grid.interp(param['wet'], 'Y'))
        param['wet_c'] = discard_land(grid.interp(param['wet'], ['X', 'Y']))
        
        ##################### Coarsegraining velocities ########################
        
        data = xr.Dataset()
        # Create coarse version of the dataset
        ds_coarse = DatasetCM26(data, param, grid)
        ds_coarse.factor = factor
        
        # Coarsegrain velocities
        ds_coarse.data['u'], ds_coarse.data['v'] = operator(self.data.u, self.data.v, self, ds_coarse)
  
        return ds_coarse

    def subgrid_forcing(self, ds_coarse, operator=operator_coarsen):
        hires_advection = self.state.advection()
        advx, advy = operator(hires_advection[0], hires_advection[1], self, ds_coarse)
        
        coarse_advection = ds_coarse.state.advection()
        SGSx = advx - coarse_advection[0]
        SGSy = advy - coarse_advection[1]
        return {'SGSx': SGSx, 'SGSy': SGSy}
    
    def sample_epoch(self, time=None, cftime=None, factors = [2,4,6,10,20], operator=operator_coarsen, percentile=1):
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
        snapshot = DatasetCM26(data.compute(), self.param, self.grid)
        
        hires_advection = snapshot.state.advection()
        advx = hires_advection[0].compute()
        advy = hires_advection[1].compute()
        
        output = {}
        for factor in factors:
            ds_coarse = snapshot.coarsen(factor, operator=operator, percentile=percentile)
            coarse_advection = ds_coarse.state.advection()
            
            ds_coarse.data['SGSx'], ds_coarse.data['SGSy'] = operator(advx, advy, snapshot, ds_coarse)
            ds_coarse.data['SGSx'] = ds_coarse.data['SGSx'] - coarse_advection[0]
            ds_coarse.data['SGSy'] = ds_coarse.data['SGSy'] - coarse_advection[1]
            
            output[factor] = ds_coarse
                            
        return output        