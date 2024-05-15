import xarray as xr
from xgcm import Grid
import numpy as np
from helpers.state_functions import StateFunctions
from helpers.operators import Coarsen, CoarsenWeighted, CoarsenKochkov, Subsampling, Filtering
from functools import cache

######## Precomputed training datasets ############
def read_datasets(keys=['train', 'test', 'validate'], factors=[4, 9, 12, 15]):
    dictionary = {}
    depth_selector = lambda x: x.isel(zl=np.arange(0,50,5))
    for factor in factors:
        base_path = f'/scratch/pp2681/mom6/CM26_datasets/ocean3d/Gauss-FGR3/factor-{factor}'
        param = depth_selector(xr.open_dataset(f'{base_path}/param.nc'))

        nfiles = {'train': 96, 'test': 24, 'validate': 12}
        for key in keys:
            file_list = [f'{base_path}/{key}-{j}.nc' for j in range(nfiles[key])]
            print('Reading from folder', base_path)
            data = xr.open_mfdataset(file_list, chunks={'zl':1, 'time':1}, concat_dim='time', combine='nested')
            dictionary[f'{key}-{factor}'] = DatasetCM26(data, param)
    return dictionary

def mask_from_nans(variable):
    mask = np.logical_not(np.isnan(variable)).astype('float32')
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

def create_grid(param):
    '''
    Depending on the dataset (2D or 3D), 
    return different grid object
    '''
    if 'zl' not in param.dims:
        grid = Grid(param, coords={
            'X': {'center': 'xh', 'right': 'xq'},
            'Y': {'center': 'yh', 'right': 'yq'}
        },
        boundary={"X": 'periodic', 'Y': 'fill'},
        fill_value = {'Y':0})
    else:
        grid = Grid(param, coords={
            'X': {'center': 'xh', 'right': 'xq'},
            'Y': {'center': 'yh', 'right': 'yq'},
            'Z': {'center': 'zl', 'outer': 'zi'}
        },
        boundary={"X": 'periodic', 'Y': 'fill', 'Z': 'fill'},
        fill_value = {'Y': 0, 'Z': 0})
    return grid

class DatasetCM26():
    def from_cloud(self, source='cmip6', compute_param=True):
        '''
        Algorithm:
        * Initialize data and grid information from cloud in a lazy way
        * Create C-Awakawa grid
        * Interpolate data to C-Arakawa grid
        * Hint: put wall to the north pole for simplicity
        '''
        ############ Read datasets ###########
        rename_surf = {'usurf': 'u', 'vsurf': 'v', 'surface_salt': 'salt', 'surface_temp': 'temp'}
        if source == 'leap':
            from intake import open_catalog
            ds = xr.open_dataset("gs://leap-persistent-ro/groundpepper/GFDL_cm2.6/GFDL_CM2_6_CONTROL_DAILY_SURF.zarr", engine='zarr', chunks={}, use_cftime=True).rename(**rename_surf)
            cat = open_catalog("https://raw.githubusercontent.com/pangeo-data/pangeo-datastore/master/intake-catalogs/ocean/GFDL_CM2.6.yaml")
            param_init  = cat["GFDL_CM2_6_grid"].to_dask()
        elif source == 'cmip6':
            ds = xr.open_dataset("gs://cmip6/GFDL_CM2_6/control/surface", engine='zarr', chunks={}, use_cftime=True).rename(**rename_surf)
            param_init = xr.open_dataset('gs://cmip6/GFDL_CM2_6/grid', engine='zarr')
        elif source == 'cmip6-3d':
            ds = xr.open_dataset("gs://cmip6/GFDL_CM2_6/control/ocean_3d", engine='zarr', chunks={}, use_cftime=True).rename(
                {'st_ocean': 'zl'})
            param_init = xr.open_dataset('gs://cmip6/GFDL_CM2_6/grid', engine='zarr').rename(
                {'st_ocean': 'zl', 'st_edges_ocean': 'zi'})
        elif '3d-' in source:
            base_path = '/scratch/pp2681/mom6/CM26_datasets/ocean3d/rawdata'
            param = xr.open_dataset(f'{base_path}/param.nc')
            if source == '3d-train':
                file_list = [f'{base_path}/train-{j}.nc' for j in range(96)]
            elif source == '3d-test':
                file_list = [f'{base_path}/test-{j}.nc' for j in range(24)]
            elif source == '3d-validate':
                file_list = [f'{base_path}/validate-{j}.nc' for j in range(12)]
            else:
                print('Error: wrong source parameter')
            data = xr.open_mfdataset(file_list, chunks={'zl':1, 'time':1}, concat_dim='time', combine='nested')

            if compute_param:
                param = param.compute()
                param = param.chunk()
            return data, param    
        else:
            print('Error: wrong source parameter')
        
        ############ Rename coordinates ###########
        rename = {'xt_ocean': 'xh', 'yt_ocean': 'yh', 'xu_ocean': 'xq', 'yu_ocean': 'yq'}
        rename_param = {'dxt': 'dxT', 'dyt': 'dyT', 'dxu': 'dxBu', 'dyu': 'dyBu'}
 
        ds = ds.rename(**rename).chunk({'yh':-1, 'yq':-1})
        param_init = param_init.rename(**rename, **rename_param).chunk({'yh':-1, 'yq':-1})

        ############ Drop unnecessary coordinates ###########
        param = xr.Dataset()
        for key in ['xh', 'yh', 'xq', 'yq', 'zl', 'zi']:
            if key in param_init.keys():
                param[key] = param_init[key]
        for key in ['dxT', 'dyT']:
            param[key] = param_init[key].drop(['area_t', 'dxT', 'dyT', 'geolat_t', 'geolon_t', 'ht', 'kmt', 'wet'])
        for key in ['dxBu', 'dyBu']:
            param[key] = param_init[key].drop(['area_u', 'dxBu', 'dyBu', 'geolat_c', 'geolon_c', 'hu', 'kmu'])
        
        ############ Init xgcm.Grid object for C-grid ###########
        # Note, we implement B.C. only in zonal diretion,
        # but simply set zero B.C. in meridional direction 
        grid = create_grid(param)
        
        ############ Compute masks for C-grid ###########
        # Note, we assume that coastline goes accross U,V and corner points,
        # while land points are uniquely defined by the T points
        param['wet'] = mask_from_nans(ds.temp)

        # Set manually wall for the layer of points
        # close to the north (and southern) pole
        param['wet'][{'yh':-1}] = 0
        param['wet'][{'yh': 0}] = 0

        # Interpolating mask from cell center to corners or sides
        # will result to values less than 1, and thus we mark
        # these points as land
        param['wet_u'] = discard_land(grid.interp(param['wet'], 'X'))
        param['wet_v'] = discard_land(grid.interp(param['wet'], 'Y'))
        param['wet_c'] = discard_land(grid.interp(param['wet'], ['X', 'Y']))
        # Mask on the vertical interface of grid cells
        if 'zl' in param.dims:
            param['wet_w'] = discard_land(grid.interp(param['wet'].chunk({'zl':-1}), 'Z'))
        
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
        
        ########### Interpolate velocities to C-grid ############
        data = xr.Dataset()
        data['u'] = grid.interp(ds.u.fillna(0.) * param.wet_c,'Y') * param.wet_u
        data['v'] = grid.interp(ds.v.fillna(0.) * param.wet_c,'X') * param.wet_v
        data['temp'] = ds.temp.fillna(0.)
        data['salt'] = ds.salt.fillna(0.)
        data['time'] = ds['time']
        
        if compute_param:
            param = param.compute()
            param = param.chunk()

        return data, param
        
    def __init__(self, data=None, param=None, source='cmip6', compute_param=True):
        if data is None or param is None:
            self.data, self.param = self.from_cloud(source=source, compute_param=compute_param)
        else:
            self.data = data
            self.param = param
        
        self.grid = create_grid(self.param)

        self.state = StateFunctions(self.data, self.param, self.grid)
        return
    
    def __del__(self):
        del self.data, self.param, self.grid, self.state
        return
    
    def nanvar(self, x):
        if 'xh' in x.dims and 'yh' in x.dims:
            return x.where(self.param.wet)
        if 'xh' in x.dims and 'yq' in x.dims:
            return x.where(self.param.wet_v)
        if 'xq' in x.dims and 'yh' in x.dims:
            return x.where(self.param.wet_u)
        if 'xq' in x.dims and 'yq' in x.dims:
            return x.where(self.param.wet_c)
    
    def select2d(self, time = None, zl=None):
        data = self.data
        param = self.param

        if 'time' in self.data.dims:
            if time is None:
                time = np.random.randint(0,len(self.data.time))
            data = data.isel(time=time)
        
        if 'zl' in self.data.dims:
            if zl is None:
                zl = np.random.randint(0,len(self.data.zl))
            data = data.isel(zl=zl)
            param = param.isel(zl=zl)
            
        return DatasetCM26(data, param)
    
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

        if 'zl' in self.param:
            param['zl'] = self.param['zl']
            param['zi'] = self.param['zi']
        
        ############ Creating xgcm.Grid object ############
        grid = create_grid(param)
        
        ######################### Creating wet masks ###########################
        param['wet'] = discard_land(self.param.wet.coarsen({'xh':factor,'yh':factor}).mean(), percentile=percentile)
        # Set manually wall for the layer of points
        # close to the north (and southern) pole
        param['wet'][{'yh':-1}] = 0
        param['wet'][{'yh': 0}] = 0
        param['wet_u'] = discard_land(grid.interp(param['wet'], 'X'))
        param['wet_v'] = discard_land(grid.interp(param['wet'], 'Y'))
        param['wet_c'] = discard_land(grid.interp(param['wet'], ['X', 'Y']))
        # Mask on the vertical interface of grid cells
        if 'zl' in param.dims:
            param['wet_w'] = discard_land(grid.interp(param['wet'].chunk({'zl':-1}), 'Z'))

        return param.compute().chunk()

    def coarsen(self, factor=10, FGR_absolute=None, FGR_multiplier=None,
                coarsening=CoarsenWeighted(), filtering=Filtering(), percentile=0):
        '''
        Coarsening of the dataset with a given factor

        Algorithm:
        * Initialize coarse grid
        * Coarsegrain velocities by applying operator
        * Return new dataset with coarse velocities

        Note: FGR is an absolute value w.r.t. fine grid
              FGR_multiplier is w.r.t. coarse grid
        '''
        FGR = None
        if FGR_multiplier is not None and FGR_absolute is not None:
            raise Exception("Provide FGR or FGR_multiplier but not both")
        if FGR_multiplier is not None:
            FGR = FGR_multiplier * factor
        if FGR_absolute is not None:
            FGR = FGR_absolute

        # Filter if needed
        if FGR is not None:
            data = xr.Dataset()
            data['u'], data['v'], data['rho'] = \
                filtering(self.data.u, self.data.v, self.state.rho(), self,
                            FGR) # Here FGR is w.r.t. fine grid
            ds_filter = DatasetCM26(data, self.param)
        else:
            ds_filter = self
        
        # Coarsegrain if needed
        if factor > 1:
            param = self.init_coarse_grid(factor=factor, percentile=percentile)
            data = xr.Dataset()
            ds_coarse = DatasetCM26(data, param)

            data['u'], data['v'], data['rho'] = \
                coarsening(ds_filter.data.u, ds_filter.data.v, ds_filter.state.rho(), 
                           ds_filter, ds_coarse, factor=factor)
            # To properly initialize ds_coarse.state
            del ds_coarse
            ds_coarse = DatasetCM26(data, param)
        else:
            ds_coarse = ds_filter
        
        return ds_coarse

    def compute_subgrid_forcing(self, factor=4, FGR_multiplier=None,
                coarsening=CoarsenWeighted(), filtering=Filtering(), percentile=0):
        '''
        This function computes the subgrid forcing, that is
        SGSx = filter(advection) - advection(coarse_state),
        for a given coarsegraining factor and coarsegraining operator.
        And returns the xr.Dataset() with lazy computations
        '''

        # Advection in high resolution model
        hires_advection = self.state.advection()

        # Coarsegrained state
        ds_coarse = self.coarsen(factor=factor, FGR_multiplier=FGR_multiplier,
                                 coarsening=coarsening, filtering=filtering,
                                 percentile=percentile) 

        # Compute advection on a coarse grid
        coarse_advection = ds_coarse.state.advection()

        # Compute subgrid forcing
        if FGR_multiplier is not None:
            hires_advection = filtering(hires_advection[0], hires_advection[1], None,
                                                        self, FGR_multiplier * factor)
        advx_coarsen, advy_coarsen, _ = coarsening(hires_advection[0], hires_advection[1], None,
                                                self, ds_coarse, factor)

        ds_coarse.data['SGSx'] = advx_coarsen - coarse_advection[0]
        ds_coarse.data['SGSy'] = advy_coarsen - coarse_advection[1]

        return ds_coarse

    def compute_subfilter_forcing(self, factor=4, FGR_multiplier=2,
                coarsening=CoarsenWeighted(), filtering=Filtering(), percentile=0):
        '''
        As compared to the "compute_subgrid_forcing" function, 
        here we evaluate contribution of subfilter stresses 
        on the grid of high resolution model. Then, coarsegraining 
        operator is used mostly for subsampling of data. An advantage of
        this method is that it is agnostic to the numerical discretization
        scheme of the advection operator.
        '''

        # Advection in high resolution model
        hires_advection = self.state.advection()

        # Filtered and filtered-coarsegrained states
        ds_filter = self.coarsen(factor=1, FGR_absolute=factor*FGR_multiplier,
                                 filtering = filtering)
        ds_coarse = ds_filter.coarsen(factor=factor, coarsening=coarsening, percentile=percentile)

        # Compute advection on a filtered state
        filter_advection = ds_filter.state.advection()

        # Filtering of the high-resolution advection
        advx_filtered, advy_filtered, _ = filtering(hires_advection[0], hires_advection[1], None,
                                                    ds_filter, FGR_multiplier * factor)

        # Subfilter forcing on a fine grid
        SGSx = advx_filtered - filter_advection[0]
        SGSy = advy_filtered - filter_advection[1]

        # Coarsegraining the subfilter forcing
        ds_coarse.data['SGSx'], ds_coarse.data['SGSy'], _ = coarsening(SGSx, SGSy, None, 
                                                                       self, ds_coarse, factor)

        return ds_coarse

    def perturb_velocities(self, grid_harmonic='plane_wave', amp=1e-3):
        '''
        As compared to the function sample_grid_harmonic, this one
        perturbs velocities

        Return new object StateFunctions with data containing waves
        '''

        perturbation = self.state.sample_grid_harmonic(grid_harmonic)
        data = self.data.copy()
        data['u'] = data['u'] + amp * perturbation.data['u']
        data['v'] = data['v'] + amp * perturbation.data['v']
        return DatasetCM26(data, self.param, self.grid)

    def predict_ANN(self, ann_Txy, ann_Txx_Tyy, ann_Tall, **kw):
        '''
        This function makes ANN inference on the whole dataset
        '''

        data = self.data[['SGSx', 'SGSy', 'u', 'v']]
        data['ZB20u'] = xr.zeros_like(data.SGSx)
        data['ZB20v'] = xr.zeros_like(data.SGSy)

        for time in range(len(self.data.time)):
            for zl in range(len(self.data.zl)):
                batch = self.select2d(time=time,zl=zl)
                prediction = batch.state.ANN(ann_Txy, ann_Txx_Tyy, ann_Tall, **kw)
                data['ZB20u'][{'time':time, 'zl':zl}] = prediction['ZB20u']
                data['ZB20v'][{'time':time, 'zl':zl}] = prediction['ZB20v']
        
        return DatasetCM26(data, self.param)
    
    def SGS_skill(self):
        '''
        This function computes:
        * 2D map of R-squared
        * 2D map of SGS dissipation
        * Power and energy transfer spectra
        in a few regions
        '''
        grid = self.grid
        param = self.param
        SGSx = self.data.SGSx
        SGSy = self.data.SGSy
        ZB20u = self.data.ZB20u
        ZB20v = self.data.ZB20v

        ############# R-squared and correlation ##############
        # Here we define second moments
        def M2(x,y=None,centered=False,dims=None,exclude_dims='zl'):
            if dims is None and exclude_dims is not None:
                dims = []
                for dim in x.dims:
                    if dim not in exclude_dims:
                        dims.append(dim)

            if y is None:
                y = x
            if centered:
                return (x*y).mean(dims) - x.mean(dims)*y.mean(dims)
            else:
                return (x*y).mean(dims)

        def M2u(x,y=None,centered=False,dims='time'):
            return grid.interp(M2(x,y,centered,dims),'X')
        def M2v(x,y=None,centered=False,dims='time'):
            return grid.interp(M2(x,y,centered,dims),'Y')
            
        errx = SGSx - ZB20u
        erry = SGSy - ZB20v

        skill = xr.Dataset()
        ######## Simplest statistics ##########
        skill['SGSx_mean'] = SGSx.mean('time')
        skill['SGSy_mean'] = SGSy.mean('time')
        skill['ZB20u_mean'] = ZB20u.mean('time')
        skill['ZB20v_mean'] = ZB20v.mean('time')
        skill['SGSx_std']  = SGSx.std('time')
        skill['SGSy_std']  = SGSy.std('time')
        skill['ZB20u_std'] = ZB20u.std('time')
        skill['ZB20v_std'] = ZB20v.std('time')

        # These metrics are same as in GZ21 work
        # Note: eveything is uncentered
        skill['R2u_map'] = 1 - M2u(errx) / M2u(SGSx)
        skill['R2v_map'] = 1 - M2v(erry) / M2v(SGSy)
        skill['R2_map']  = 1 - (M2u(errx) + M2v(erry)) / (M2u(SGSx) + M2v(SGSy))

        # Here everything is centered according to definition of correlation
        skill['corru_map'] = M2u(SGSx,ZB20u,centered=True) / np.sqrt(M2u(SGSx,centered=True) * M2u(ZB20u,centered=True))
        skill['corrv_map'] = M2v(SGSy,ZB20v,centered=True) / np.sqrt(M2v(SGSy,centered=True) * M2v(ZB20v,centered=True))
        # It is complicated to derive a single true formula, so use simplest one
        skill['corr_map']  = (skill['corru_map'] + skill['corrv_map']) * 0.5

        ########### Global metrics ############
        skill['R2u'] = 1 - M2(errx) / M2(SGSx)
        skill['R2v'] = 1 - M2(erry) / M2(SGSy)
        skill['R2'] = 1 - (M2(errx) + M2(erry)) / (M2(SGSx) + M2(SGSy))
        skill['corru'] = M2(SGSx,ZB20u,centered=True) \
            / np.sqrt(M2(SGSx,centered=True) * M2(ZB20u,centered=True))
        skill['corrv'] = M2(SGSy,ZB20v,centered=True) \
            / np.sqrt(M2(SGSy,centered=True) * M2(ZB20v,centered=True))
        skill['corr'] = (skill['corru'] + skill['corrv']) * 0.5
        skill['opt_scaling'] = (M2(SGSx,ZB20u) + M2(SGSy,ZB20v)) / (M2(ZB20u) + M2(ZB20v))

        ############### Spectral analysis ##################
        for region in ['NA', 'Pacific', 'Equator', 'ACC']:
            transfer, power, KE_spec, power_time, KE_time = self.state.transfer(SGSx, SGSy, region=region, additional_spectra=True)
            skill['transfer_'+region] = transfer.rename({'freq_r': 'freq_r_'+region})
            skill['power_'+region] = power.rename({'freq_r': 'freq_r_'+region})
            skill['KE_spec_'+region] = KE_spec.rename({'freq_r': 'freq_r_t'+region})
            skill['power_time_'+region] = power_time
            skill['KE_time_'+region] = KE_time
            transfer, power, KE_spec, power_time, KE_time = self.state.transfer(ZB20u, ZB20v, region=region, additional_spectra=True)
            skill['transfer_ZB_'+region] = transfer.rename({'freq_r': 'freq_r_'+region})
            skill['power_ZB_'+region] = power.rename({'freq_r': 'freq_r_'+region})
            skill['power_time_ZB_'+region] = power_time

        return skill.compute()