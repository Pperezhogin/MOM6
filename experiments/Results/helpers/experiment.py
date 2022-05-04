import xarray as xr
import os
import numpy as np

# Imitates xarray. All variables are
# returned as @property. Compared to xarray, allows
# additional computational tools and initialized instantly (within ms)
class Experiment:
    def __init__(self, folder):
        '''
        Initializes with folder containing all experiments.
        Xarray datasets are read only by demand within @property function

        All fields needed for plotting snapshots must be registered
        as properties
        '''
        self.folder = folder

        if not os.path.exists(os.path.join(self.folder, 'ocean_geometry.nc')):
            print('Error, cannot find files in folder'+self.folder)

        # xarray dataset. Private variable, access by @property function
        self.__param = None
        self.__series = None
        self.__ave = None
        self.__prog = None
        self.__energy = None
        self.__forcing = None
        self.__mom = None

    ########################## Service functions #############################
    def rename_coordinates(self, xr_dataset):
        '''
        in-place change of coordinate names to Longitude and Latitude.
        For simplicity of plotting with xarray.plot()
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
    @property
    def param(self):
        if self.__param is None:
            self.__param = xr.open_dataset(os.path.join(self.folder, 'ocean_geometry.nc')).rename(
                {'latq': 'yq', 'lonq': 'xq', 'lath': 'yh', 'lonh': 'xh'} # change coordinates notation as in other files
            )
            self.rename_coordinates(self.__param)
        return self.__param
    
    @property
    def series(self):
        if self.__series is None:
            self.__series = xr.open_dataset(os.path.join(self.folder, 'ocean.stats.nc'), decode_times=False)
        return self.__series

    @property
    def ave(self):
        if self.__ave is None:
            self.__ave = xr.open_mfdataset(os.path.join(self.folder, 'ave_*.nc'), decode_times=False, concat_dim='Time', parallel=True)
            self.rename_coordinates(self.__ave)
        return self.__ave

    @property
    def prog(self):
        if self.__prog is None:
            self.__prog = xr.open_mfdataset(os.path.join(self.folder, 'prog_*.nc'), decode_times=False, concat_dim='Time', parallel=True)
            self.rename_coordinates(self.__prog)
        return self.__prog
    
    @property
    def energy(self):
        if self.__energy is None:
            self.__energy = xr.open_mfdataset(os.path.join(self.folder, 'energy_*.nc'), decode_times=False, concat_dim='Time', parallel=True)
            self.rename_coordinates(self.__energy)
        return self.__energy

    @property
    def forcing(self):
        if self.__forcing is None:
            self.__forcing = xr.open_mfdataset(os.path.join(self.folder, 'forcing_*.nc'), decode_times=False, concat_dim='Time', parallel=True)
            self.rename_coordinates(self.__forcing)
        return self.__forcing

    @property
    def mom(self):
        if self.__mom is None:
            self.__mom = xr.open_mfdataset(os.path.join(self.folder, 'mom_*.nc'), decode_times=False, concat_dim='Time', parallel=True)
            self.rename_coordinates(self.__mom)
        return self.__mom

    ##################### Analog of xarray. Set variables ####################
    # It is good to define these @property fields to know which
    # variables will be used in computations
    @property
    def RV_f(self):
        RV = self.prog.RV
        f = self.param.f
        return RV/f
    
    @property
    def PV(self):
        return self.prog.PV

    @property
    def KE(self):
        return self.energy.KE

    @property
    def smagx(self):
        return self.mom.diffu-self.mom.ZB2020u

    @property
    def smagy(self):
        return self.mom.diffv-self.mom.ZB2020v

    @property
    def ZB2020u(self):
        return self.mom.ZB2020u

    @property
    def ZB2020v(self):
        return self.mom.ZB2020v

    @property
    def e(self):
        return self.prog.e

    ######################## Coarsening tools ########################