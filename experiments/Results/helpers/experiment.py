import xarray as xr
import os
import numpy as np
from functools import cached_property

# Imitates xarray. All variables are
# returned as @property. Compared to xarray, allows
# additional computational tools and initialized instantly (within ms)
class Experiment:
    def __init__(self, folder):
        '''
        Initializes with folder containing all experiments.
        Xarray datasets are read only by demand within @property function
        @cached_property allows to read each netcdf file only ones

        All fields needed for plotting snapshots must be registered
        as properties
        '''
        self.folder = folder

        if not os.path.exists(os.path.join(self.folder, 'ocean_geometry.nc')):
            print('Error, cannot find files in folder'+self.folder)

    ########################## Service functions #############################
    def rename_coordinates(self, xr_dataset):
        '''
        in-place change of coordinate names to Longitude and Latitude.
        For convenience of plotting with xarray.plot()
        '''
        for key in ['xq', 'xh']:
            try:
                xr_dataset[key].attrs['long_name'] = 'Longitude'
                xr_dataset[key].attrs['units'] = ''
            except:
                pass

        for key in ['yq', 'yh']:
            try:
                xr_dataset[key].attrs['long_name'] = 'Latitude'
                xr_dataset[key].attrs['units'] = ''
            except:
                pass
    
    ################### Getters for netcdf files as xarrays #####################
    @cached_property
    def series(self):
        result = xr.open_dataset(os.path.join(self.folder, 'ocean.stats.nc'), decode_times=False)
        return result

    @cached_property
    def param(self):
        result = xr.open_dataset(os.path.join(self.folder, 'ocean_geometry.nc')).rename(
                {'latq': 'yq', 'lonq': 'xq', 'lath': 'yh', 'lonh': 'xh'} # change coordinates notation as in other files
            )
        self.rename_coordinates(result)
        return result

    @cached_property
    def ave(self):
        result = xr.open_mfdataset(os.path.join(self.folder, 'ave_*.nc'), decode_times=False, concat_dim='Time', parallel=True)
        self.rename_coordinates(result)
        return result

    @cached_property
    def prog(self):
        result = xr.open_mfdataset(os.path.join(self.folder, 'prog_*.nc'), decode_times=False, concat_dim='Time', parallel=True)
        self.rename_coordinates(result)
        return result
    
    @cached_property
    def energy(self):
        result = xr.open_mfdataset(os.path.join(self.folder, 'energy_*.nc'), decode_times=False, concat_dim='Time', parallel=True)
        self.rename_coordinates(result)
        return result

    @cached_property
    def forcing(self):
        result = xr.open_mfdataset(os.path.join(self.folder, 'forcing_*.nc'), decode_times=False, concat_dim='Time', parallel=True)
        self.rename_coordinates(result)
        return result

    @cached_property
    def mom(self):
        result = xr.open_mfdataset(os.path.join(self.folder, 'mom_*.nc'), decode_times=False, concat_dim='Time', parallel=True)
        self.rename_coordinates(result)
        return result

    ##################### Analog of xarray. Set variables ####################
    # It is good to define these @property fields to know which
    # variables will be used in computations
    @cached_property
    def RV(self):
        return self.prog.RV

    @cached_property
    def RV_f(self):
        return self.RV / self.param.f

    @cached_property
    def PV(self):
        return self.prog.PV

    @cached_property
    def e(self):
        return self.prog.e

    @cached_property
    def u(self):
        return self.prog.u

    @cached_property
    def v(self):
        return self.prog.v

    ########### Auxiliary variables. Not involved in coarsegraining ##########
    @cached_property
    def KE(self):
        return self.energy.KE

    @cached_property
    def smagx(self):
        return self.mom.diffu-self.mom.ZB2020u

    @cached_property
    def smagy(self):
        return self.mom.diffv-self.mom.ZB2020v

    @cached_property
    def ZB2020u(self):
        return self.mom.ZB2020u

    @cached_property
    def ZB2020v(self):
        return self.mom.ZB2020v