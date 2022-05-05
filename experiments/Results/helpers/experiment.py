import xarray as xr
import os
import numpy as np
import xrft
from functools import cached_property
from helpers.computational_tools import remesh, compute_2dfft
# Imitates xarray. All variables are
# returned as @property. Compared to xarray, allows
# additional computational tools and initialized instantly (within ms)
class Experiment:
    def __init__(self, folder):
        '''
        Initializes with folder containing all netcdf files corresponding
        to a given experiment.
        Xarray datasets are read only by demand within @property decorator
        @cached_property allows to read each netcdf file only ones

        All fields needed for plotting purposes are suggested to be
        registered with @cached_property decorator (for convenience)
        '''
        self.folder = folder

        if folder != 'This is detached experiment':
            if not os.path.exists(os.path.join(self.folder, 'ocean_geometry.nc')):
                print('Error, cannot find files in folder'+self.folder)
    
    def remesh(self, target, compute=False):
        '''
        Returns object "experiment", where "Main variables"
        are coarsegrained according to resolution of the target experiment
        '''

        # The coarsegrained experiment is no longer attached to the folder
        result = Experiment(folder='This is detached experiment')

        # Coarsegrain "Main variables" explicitly
        for key in ['RV', 'RV_f', 'PV', 'e', 'u', 'v']:
            if compute:
                setattr(result, key, remesh(self.__getattribute__(key),target.__getattribute__(key)).compute())
            else:
                setattr(result, key, remesh(self.__getattribute__(key),target.__getattribute__(key)))

        result.param = target.param # copy coordinates from target experiment

        return result        

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

    ############################### Main variables  #########################
    # These variables are suggested to be used in computations, i.e. coarsegraining,
    # spectra and so on. They are recommended to be registered
    # The use of @property decorator allows not to load data at initialization
    # They must be registered in "coarsegrain" method
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

    ########### Auxiliary variables. Not involved in computations ##########
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

    ############## Computational tools. Spectra, and so on #################
    @cached_property
    def KE(self):
        return 0.5 * (remesh(self.u**2, self.e) + remesh(self.v**2, self.e))

    @cached_property
    def fft_u(self):
        return compute_2dfft(remesh(self.u, self.e), self.param.dxT, self.param.dyT, Lat=(35,45), Lon=(5,15), window='hann')

    @cached_property
    def fft_v(self):
        return compute_2dfft(remesh(self.v, self.e), self.param.dxT, self.param.dyT, Lat=(35,45), Lon=(5,15), window='hann')

    @cached_property
    def KE_spectrum(self):
        return xrft.isotropize((np.abs(self.fft_u)**2+np.abs(self.fft_v)**2)/2, fftdim=('kx','ky'), nfactor=2, truncate=True)