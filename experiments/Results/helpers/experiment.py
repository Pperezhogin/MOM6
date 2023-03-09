import xarray as xr
import os
import numpy as np
import xrft
from functools import cached_property
from helpers.computational_tools import rename_coordinates, remesh, compute_isotropic_KE, compute_isotropic_cospectrum, compute_isotropic_PE, compute_KE_time_spectrum, mass_average, Lk_error, select_LatLon, diffx_uq, diffy_vq, diffx_tv, diffy_tu, prodx_uq, prody_vq, filter_iteration
from helpers.netcdf_cache import netcdf_property

Averaging_Time = slice(3650,7300)
class main_property(cached_property):
    '''
    https://stackoverflow.com/questions/9541025/how-to-copy-a-python-class
    '''
    pass

class Experiment:
    '''
    Imitates xarray. All variables are
    returned as @property. Compared to xarray, allows
    additional computational tools and initialized instantly (within ms)
    '''
    def __init__(self, folder, key=''):
        '''
        Initializes with folder containing all netcdf files corresponding
        to a given experiment.
        Xarray datasets are read only by demand within @property decorator
        @cached_property allows to read each netcdf file only ones

        All fields needed for plotting purposes are suggested to be
        registered with @cached_property decorator (for convenience)
        '''
        self.folder = folder
        self.key = key # for storage of statistics
        self.recompute = False # default value of recomputing of cached on disk properties

        if not os.path.exists(os.path.join(self.folder, 'ocean_geometry.nc')):
            print('Error, cannot find files in folder'+self.folder)

    def remesh(self, target, key, compute=False):
        '''
        Returns object "experiment", where "Main variables"
        are coarsegrained according to resolution of the target experiment
        '''

        # The coarsegrained experiment is no longer attached to the folder
        result = Experiment(folder=self.folder, key=key)

        # Coarsegrain "Main variables" explicitly
        for key in Experiment.get_list_of_main_properties():
            if compute:
                setattr(result, key, remesh(self.__getattribute__(key),target.__getattribute__(key)).compute())
            else:
                setattr(result, key, remesh(self.__getattribute__(key),target.__getattribute__(key)))

        result.param = target.param # copy coordinates from target experiment
        result._hires = self # store reference to the original experiment

        return result

    def Lk_error(self, target_exp, features=['MKE_val'], k=2):
        '''
        Computes averaged over characteristics
        normalized L1 error. Characteristics at each
        layer are considered to be independent (i.e. they are averaged)

        terget_exp - instance of Experiment (reference simulation)
        '''
        errors_list = []
        errors_dict = {}

        for feature in features:
            input = self.__getattribute__(feature)
            target = target_exp.__getattribute__(feature)
            error = Lk_error(input, target, k=k)
            errors_list.extend(error)
            errors_dict[feature] = error

        return errors_list, errors_dict

    @classmethod
    def get_list_of_netcdf_properties(cls):
        '''
        Allows to know what properties should be cached
        https://stackoverflow.com/questions/27503965/list-property-decorated-methods-in-a-python-class
        '''
        result = []
        for name, value in vars(cls).items():
            if isinstance(value, netcdf_property):
                result.append(name)
        return result

    @classmethod
    def get_list_of_main_properties(cls):
        '''
        Allows to know what properties should be coarsegrained
        '''
        result = []
        for name, value in vars(cls).items():
            if isinstance(value, main_property):
                result.append(name)
        return result
    
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
        rename_coordinates(result)
        return result

    @cached_property
    def vert_grid(self):
        return xr.open_dataset(os.path.join(self.folder, 'Vertical_coordinate.nc')).rename({'Layer': 'zl'})

    @cached_property
    def prog(self):
        result = xr.open_mfdataset(os.path.join(self.folder, 'prog_*.nc'), decode_times=False, parallel=True, chunks={'Time': 5, 'zl': 2})
        rename_coordinates(result)
        return result
    
    @cached_property
    def mom(self):
        result = xr.open_mfdataset(os.path.join(self.folder, 'mom_*.nc'), decode_times=False, parallel=True, chunks={'Time': 5, 'zl': 2})
        rename_coordinates(result)
        return result

    @cached_property
    def energy(self):
        result = xr.open_mfdataset(os.path.join(self.folder, 'energy_*.nc'), decode_times=False, parallel=True, chunks={'Time': 5, 'zl': 2})
        rename_coordinates(result)
        return result

    @cached_property
    def ave(self):
        result = xr.open_mfdataset(os.path.join(self.folder, 'ave_*.nc'), decode_times=False, parallel=True, chunks={'Time': 5, 'zl': 2})
        rename_coordinates(result)
        return result

    @cached_property
    def forcing(self):
        return xr.open_mfdataset(os.path.join(self.folder, 'forcing_*.nc'), decode_times=False)
    
    ############################### Main variables  #########################
    # These variables are used in statistical tools. They will be coarsegrained
    # with remesh function
    @main_property
    def u(self):
        return self.prog.u

    @main_property
    def v(self):
        return self.prog.v
    
    @main_property
    def e(self):
        return self.prog.e

    @main_property
    def h(self):
        return self.prog.h

    @main_property
    def ua(self):
        return self.ave.u

    @main_property
    def va(self):
        return self.ave.v

    @main_property
    def ea(self):
        return self.ave.e

    @main_property
    def ha(self):
        return self.ave.h

    ######################## Auxiliary variables #########################
    @main_property
    def RV(self):
        return self.prog.RV

    @main_property
    def RV_f(self):
        return self.RV / self.param.f

    @main_property
    def PV(self):
        return self.prog.PV
    
    @property
    def smagu(self):
        return self.mom.diffu-self.mom.ZB2020u
    
    @property
    def smagv(self):
        return self.mom.diffv-self.mom.ZB2020v

    ########################  Statistical tools  #########################
    #################  Express through main properties ###################

    #-------------------  Mean flow and variability  --------------------#
    @netcdf_property
    def ssh_mean(self):
        return self.ea.isel(zi=0).sel(Time=Averaging_Time).mean(dim='Time')

    @netcdf_property
    def ssh_var(self):
        return self.ea.isel(zi=0).sel(Time=Averaging_Time).var(dim='Time')

    @netcdf_property
    def u_mean(self):
        return self.ua.sel(Time=Averaging_Time).mean(dim='Time')

    @netcdf_property
    def v_mean(self):
        return self.va.sel(Time=Averaging_Time).mean(dim='Time')

    @netcdf_property
    def h_mean(self):
        return self.ha.sel(Time=Averaging_Time).mean(dim='Time')

    @netcdf_property
    def e_mean(self):
        return self.ea.sel(Time=Averaging_Time).mean(dim='Time')

    #-----------------------  Spectral analysis  ------------------------#
    @netcdf_property
    def KE_spectrum_series(self):
        return compute_isotropic_KE(self.u, self.v, self.param.dxT, self.param.dyT)

    @netcdf_property
    def KE_spectrum_global_series(self):
        return compute_isotropic_KE(self.u, self.v, self.param.dxT, self.param.dyT, 
            Lat=(30,50), Lon=(0,22))

    @netcdf_property
    def KE_spectrum(self):
        return self.KE_spectrum_series.sel(Time=Averaging_Time).mean(dim='Time')

    @netcdf_property
    def KE_spectrum_global(self):
        return self.KE_spectrum_global_series.sel(Time=Averaging_Time).mean(dim='Time')
    
    @netcdf_property
    def Smagorinsky_transfer(self):
        return compute_isotropic_cospectrum(self.u, self.v, self.smagu, self.smagv,
            self.param.dxT, self.param.dyT).sel(Time=Averaging_Time).mean(dim='Time')
    
    @netcdf_property
    def ZB_transfer(self):
        return compute_isotropic_cospectrum(self.u, self.v, self.mom.ZB2020u, self.mom.ZB2020v,
            self.param.dxT, self.param.dyT).sel(Time=Averaging_Time).mean(dim='Time')
    
    @property
    def Model_transfer(self):
        return self.Smagorinsky_transfer+self.ZB_transfer
    
    @netcdf_property
    def SGS_transfer(self):
        return compute_isotropic_cospectrum(self.u, self.v, self.SGSx, self.SGSy,
            self.param.dxT, self.param.dyT).sel(Time=Averaging_Time).mean(dim='Time')
    
    @netcdf_property
    def Smagorinsky_power(self):
        return 2*compute_isotropic_KE(self.smagu, self.smagv,
            self.param.dxT, self.param.dyT).sel(Time=Averaging_Time).mean(dim='Time')
    
    @netcdf_property
    def ZB_power(self):
        return 2*compute_isotropic_KE(self.mom.ZB2020u, self.mom.ZB2020v,
            self.param.dxT, self.param.dyT).sel(Time=Averaging_Time).mean(dim='Time')
    
    @netcdf_property
    def Model_power(self):
        return 2*compute_isotropic_KE(self.mom.diffu, self.mom.diffv,
            self.param.dxT, self.param.dyT).sel(Time=Averaging_Time).mean(dim='Time')

    @netcdf_property
    def SGS_power(self):
        return 2*compute_isotropic_KE(self.SGSx, self.SGSy,
            self.param.dxT, self.param.dyT).sel(Time=Averaging_Time).mean(dim='Time')
    
    @property
    def kmax(self):
        '''
        Nyquist wavenumber
        '''
        Lat=(35,45); Lon=(5,15)
        dx = select_LatLon(self.param.dxT,Lat,Lon).mean().values
        dy = select_LatLon(self.param.dyT,Lat,Lon).mean().values
        return np.pi/np.max([dx,dy])

    @netcdf_property
    def MKE_spectrum(self):
        return compute_isotropic_KE(self.u_mean, self.v_mean, self.param.dxT, self.param.dyT)

    @netcdf_property
    def EKE_spectrum(self):
        return self.KE_spectrum - self.MKE_spectrum

    @netcdf_property
    def PE_spectrum(self):
        H0 = 1000. # 1000 is reference depth (see series.H0.isel(Interface=1))
        hint = self.e.isel(zi=1)+H0 # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/diagnostics/MOM_sum_output.F90#L655
        mask = self.h.isel(zl=1)>1e-9 # mask of wet points. Boundaries have values 1e-10; https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/diagnostics/MOM_sum_output.F90#L656
        hint = hint * mask
        return compute_isotropic_PE(hint, self.param.dxT, self.param.dyT).sel(Time=Averaging_Time).mean(dim='Time')

    @property
    def EKE_spectrum_direct(self):
        # Difference with EKE_spectrum 0.7%
        u_eddy = self.u.sel(Time=Averaging_Time) - self.u_mean
        v_eddy = self.v.sel(Time=Averaging_Time) - self.v_mean
        return compute_isotropic_KE(u_eddy, v_eddy, self.param.dxT, self.param.dyT).mean(dim='Time')

    @netcdf_property
    def KE_time_spectrum(self):
        return compute_KE_time_spectrum(self.ua, self.va, Time=Averaging_Time)

    #-------------------------  KE, MKE, EKE  ---------------------------#        
    @netcdf_property
    def KE(self):
        return 0.5 * (remesh(self.u**2, self.e) + remesh(self.v**2, self.e))
    
    @property
    def velocity(self):
        return np.sqrt(2*self.KE)

    @netcdf_property
    def KE_series(self):
        # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/diagnostics/MOM_sum_output.F90#L763
        # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/diagnostics/MOM_sum_output.F90#L501
        # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/diagnostics/MOM_sum_output.F90#L674
        return mass_average(self.KE, self.h, self.param.dxT, self.param.dyT)
    
    @property
    def EKE_series(self):
        return (self.KE_series - self.MKE_val).sel(Time=Averaging_Time)

    @netcdf_property
    def MKE(self):
        return 0.5 * (remesh(self.u_mean**2, self.e) + remesh(self.v_mean**2, self.e))  

    @netcdf_property
    def EKE(self):
        eke = self.KE.sel(Time=Averaging_Time).mean(dim='Time') - self.MKE
        eke = eke.where(eke>0).fillna(0) # 
        return eke

    @property
    def EKE_old(self):
        try:
            eke = self.energy.KE.sel(Time=Averaging_Time).mean(dim='Time') - self.MKE
            eke = eke.where(eke>0).fillna(0) #
        except:
            eke = self.EKE
        return eke

    @netcdf_property
    def KE_total(self):
        return self.MKE+self.EKE

    @netcdf_property
    def MKE_val(self):
        return mass_average(self.MKE, self.h_mean, self.param.dxT, self.param.dyT)

    @netcdf_property
    def EKE_val(self):
        return mass_average(self.EKE, self.h_mean, self.param.dxT, self.param.dyT)

    @netcdf_property
    def KE_total_val(self):
        return self.MKE_val + self.EKE_val
    
    @netcdf_property
    def MKE_joul(self):
        return (0.5 * self.vert_grid.R * (remesh(self.u_mean**2, self.e) + remesh(self.v_mean**2, self.e)) * self.h_mean * self.param.dxT * self.param.dyT).sum(dim=('xh','yh'))
    
    @netcdf_property
    def EKE_joul(self):
        return (0.5 * self.vert_grid.R * (remesh(self.u**2, self.e) + remesh(self.v**2, self.e)) * self.h * self.param.dxT * self.param.dyT).sum(dim=('xh','yh')).sel(Time=Averaging_Time).mean(dim='Time') - self.MKE_joul
    
    #-------------------------  PE, MPE, EPE  ---------------------------#
    @netcdf_property
    def PE_Joul_series(self):
        # APE for internal interface, total value in Joules. Compare to seires.APE.isel(Interface=1)
        H0 = 1000. # 1000 is reference depth (see series.H0.isel(Interface=1))
        hint = self.e.isel(zi=1)+H0 # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/diagnostics/MOM_sum_output.F90#L655
        mask = self.h.isel(zl=1)>1e-9 # mask of wet points. Boundaries have values 1e-10; https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/diagnostics/MOM_sum_output.F90#L656
        return 0.5 * self.vert_grid.R.isel(zl=1)*self.vert_grid.g.isel(zl=1)*(hint**2 * mask * self.param.dxT * self.param.dyT).sum(dim=('xh','yh')) # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/diagnostics/MOM_sum_output.F90#L657
    
    @netcdf_property
    def PE_ssh_series(self):
        hint = self.e.isel(zi=0)
        return 0.5 * self.vert_grid.R.isel(zl=0)*self.vert_grid.g.isel(zl=0)*(hint**2 * self.param.dxT * self.param.dyT).sum(dim=('xh','yh'))
    
    @netcdf_property
    def PE(self):
        '''
        Avaliable Potential for internal interface devided by all constants, i.e. [m^2]
        '''
        H0 = 1000.
        hint = self.e.isel(zi=1)+H0
        mask = self.h.isel(zl=1)>1e-9
        hint = hint * mask
        return 0.5 * hint**2

    @netcdf_property
    def MPE(self):
        '''
        Avaliable Potential for internal interface of the mean flow [m^2]
        '''
        H0 = 1000.
        hint = self.e_mean.isel(zi=1)+H0
        mask = self.h_mean.isel(zl=1)>1e-9
        hint = hint * mask
        return 0.5 * hint**2

    @netcdf_property
    def EPE(self):
        '''
        Potential energy of the eddy fluctuations [m^2]
        '''
        epe = self.PE.sel(Time=Averaging_Time).mean(dim='Time') - self.MPE
        epe = epe.where(epe>0).fillna(0)
        return epe

    @netcdf_property
    def PE_total(self):
        return self.MPE + self.EPE

    @netcdf_property
    def MPE_val(self):
        '''
        Bad behaviour near the boundary!!!
        [m^2]
        '''
        return select_LatLon(self.MPE,Lat=(35,45), Lon=(5,15)).mean(dim=('xh','yh'))
    
    @netcdf_property
    def EPE_val(self):
        return select_LatLon(self.EPE,Lat=(35,45), Lon=(5,15)).mean(dim=('xh','yh'))

    @netcdf_property
    def PE_total_val(self):
        return self.MPE_val + self.EPE_val
    
    @netcdf_property
    def MPE_joul(self):
        '''
        MPE value, in [Joules] but not [m^2]
        '''
        H0 = 1000.
        hint = self.e_mean.isel(zi=1)+H0
        mask = self.h_mean.isel(zl=1)>1e-9
        hint = hint * mask
        return 0.5 * self.vert_grid.R.isel(zl=1)*self.vert_grid.g.isel(zl=1)*(hint**2 * mask * self.param.dxT * self.param.dyT).sum(dim=('xh','yh'))
    
    @netcdf_property
    def EPE_joul(self):
        return self.PE_Joul_series.sel(Time=Averaging_Time).mean(dim='Time') - self.MPE_joul
    
    @netcdf_property
    def MPE_ssh(self):
        hint = self.e_mean.isel(zi=0)
        return 0.5 * self.vert_grid.R.isel(zl=0)*self.vert_grid.g.isel(zl=0)*(hint**2 * self.param.dxT * self.param.dyT).sum(dim=('xh','yh'))
    
    @netcdf_property
    def EPE_ssh(self):
        return self.PE_ssh_series.sel(Time=Averaging_Time).mean(dim='Time') - self.MPE_ssh
    
    # ------------------ Advection Arakawa(gradKE)-Sadourny(PVxuv) ---------------------- #
    # https://mom6.readthedocs.io/en/dev-gfdl/api/generated/pages/Governing_Equations.html
    @property
    def KE_Arakawa(self):
        '''
        https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L1000-L1003
        '''
        areaCu = self.param.dxCu * self.param.dyCu
        areaCv = self.param.dxCv * self.param.dyCv
        areaT = self.param.dxT * self.param.dyT
        return 0.5 * (remesh(areaCu*self.u**2, areaT) + remesh(areaCv*self.v**2, areaT)) / areaT
    
    @property
    def gradKE(self):
        '''
        https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L1029-L1034
        '''
        KE = self.KE_Arakawa
        IdxCu = 1. / self.param.dxCu
        IdyCv = 1. / self.param.dyCv

        KEx = diffx_uq(KE, IdxCu) * IdxCu
        KEy = diffy_vq(KE, IdyCv) * IdyCv
        return (KEx, KEy)
    
    @property
    def relative_vorticity(self):
        '''
        https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L472
        '''
        dyCv = self.param.dyCv
        dxCu = self.param.dxCu
        IareaBu = 1. / (self.param.dxBu * self.param.dyBu)
        # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L309-L310
        dvdx = diffx_uq(self.v*dyCv,IareaBu)
        dudy = diffy_vq(self.u*dxCu,IareaBu)
        return (dvdx - dudy) * IareaBu
    
    @property
    def PV_cross_uv(self):
        '''
        https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L669-L671
        https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L788-L790
        fx = + q * vh
        fy = - q * uh
        '''
        # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L131
        # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_continuity_PPM.F90#L569-L570
        uh = self.u * remesh(self.h,self.u) * self.param.dyCu
        # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L133
        vh = self.v * remesh(self.h,self.v) * self.param.dxCv
        # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L484
        rel_vort = self.relative_vorticity

        # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L247
        Area_h = self.param.dxT * self.param.dyT
        # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L272-L273
        Area_q = remesh(Area_h, rel_vort) * 4
        # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L323
        hArea_u = remesh(Area_h*self.h,uh)
        # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L320
        hArea_v = remesh(Area_h*self.h,vh)
        # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L488
        hArea_q = 2*remesh(hArea_u,rel_vort) + 2*remesh(hArea_v,rel_vort)
        # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L489
        Ih_q = Area_q / hArea_q
        
        # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L490
        q = rel_vort * Ih_q

        IdxCu = 1. / self.param.dxCu
        IdyCv = 1. / self.param.dyCv
        # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L669-L671
        CAu = + remesh(q * remesh(vh,q),IdxCu) * IdxCu
        # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L788-L790
        CAv = - remesh(q * remesh(uh,q),IdyCv) * IdyCv

        return (CAu, CAv)
    
    @property
    def advection(self):
        '''
        https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L751
        https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L875
        '''
        CAu, CAv = self.PV_cross_uv
        KEx, KEy = self.gradKE
        return (CAu - KEx, CAv - KEy)
    
    @property
    def subgrid_forcing(self):
        '''
        self - coarsegrained experiment
        hires - highres experiment from which 
        it was coarsegrained
        '''
        if hasattr(self, '_hires'):
            adv = self.advection
            hires_advection = self._hires.advection

            fx = remesh(hires_advection[0],adv[0]) - adv[0]
            fy = remesh(hires_advection[1],adv[1]) - adv[1]

            return (fx,fy)
        else:
            print('Error: subgrid forcing cannot be computed')
            print('because there is no associated hires experiment')
            return
        
    @netcdf_property
    def SGSx(self):
        return self.subgrid_forcing[0]
    
    @netcdf_property
    def SGSy(self):
        return self.subgrid_forcing[1]
    
    # --------------------- ZB offline ---------------------- #
    @property
    def dudx(self):
        IdyCu = 1. / self.param.dyCu
        DY_dxT = self.param.dyT / self.param.dxT
        return DY_dxT * diffx_tv(self.u*IdyCu,DY_dxT)
    
    @property
    def dvdy(self):
        IdxCv = 1. / self.param.dxCv
        DX_dyT = self.param.dxT / self.param.dyT
        return DX_dyT * diffy_tu(self.v*IdxCv,DX_dyT)
    
    @property
    def dudy(self):
        IdxCu = 1. / self.param.dxCu
        DX_dyBu = self.param.dxBu / self.param.dyBu
        return DX_dyBu * diffy_vq(self.u*IdxCu,DX_dyBu)
    
    @property
    def dvdx(self):
        IdyCv = 1. / self.param.dyCv
        DY_dxBu = self.param.dyBu / self.param.dxBu
        return DY_dxBu * diffx_uq(self.v*IdyCv,DY_dxBu)
    
    @property
    def sh_xx(self):
        return self.dudx - self.dvdy
    
    @property
    def sh_xy(self):
        return self.dvdx + self.dudy
    
    @property
    def vort_xy(self):
        return self.dvdx - self.dudy
    
    @property
    def h_u(self):
        return remesh(self.h, self.u)
    
    @property
    def h_v(self):
        return remesh(self.h, self.v)
    
    @property
    def hq(self):
        h2uq = 4.0 * prody_vq(self.h_u,self.RV)
        h2vq = 4.0 * prodx_uq(self.h_v,self.RV)
        return (2. * (h2uq * h2vq)) / ((h2uq + h2vq) * (2.*remesh(self.h_u,self.RV) + 2.*remesh(self.h_v,self.RV)))
    
    def divergence(self, S_11, S_12, S_22):
        IdxCu = 1. / self.param.dxCu
        IdyCu = 1. / self.param.dyCu
        IdxCv = 1. / self.param.dxCv
        IdyCv = 1. / self.param.dyCv
        
        IareaCu = 1. / (self.param.dxCu * self.param.dyCu)
        IareaCv = 1. / (self.param.dxCv * self.param.dyCv)
        dx2q = self.param.dxBu**2
        dy2q = self.param.dyBu**2
        dx2h = self.param.dxT**2
        dy2h = self.param.dyT**2

        S_11 = S_11 * self.h
        S_22 = S_22 * self.h
        S_12 = S_12 * self.hq

        fx = (
                IdyCu * diffx_uq(dy2h*S_11,IdyCu) +
                IdxCu * diffy_tu(dx2q*S_12,IdxCu)
        ) * IareaCu / self.h_u
        
        fy = (
                IdyCv * diffx_tv(dy2q*S_12,IdyCv) + 
                IdxCv * diffy_vq(dx2h*S_22,IdxCv)
        ) * IareaCv / self.h_v

        return (fx,fy)

    def ZB_offline(self, amplitude=1./24, amp_bottom=-1, 
            ZB_type=0, ZB_cons=1, 
            LPF_iter=0, LPF_order=1,
            HPF_iter=0, HPF_order=1,
            Stress_iter=0, Stress_order=1, **kw):
        amp = xr.DataArray([amplitude, amp_bottom if amp_bottom > -0.5 else amplitude], dims=['zl'])

        areaBu = self.param.dxBu * self.param.dyBu
        areaT = self.param.dxT * self.param.dyT

        def ftr(x):
            x = filter_iteration(x,HPF_iter,HPF_order,self.h,residual=True)
            return filter_iteration(x,LPF_iter,LPF_order,self.h,residual=False)

        sh_xx = ftr(self.sh_xx)
        sh_xy = ftr(self.sh_xy)
        vort_xy = ftr(self.vort_xy)

        if ZB_type == 0:
            sum_sq = 0.5 * (remesh(vort_xy,self.h)**2+remesh(sh_xy,self.h)**2+sh_xx**2)
        elif ZB_type == 1:
            sum_sq = 0.
        
        if ZB_cons == 0:
            vort_sh = remesh(vort_xy,self.h) * remesh(sh_xy,self.h)
        elif ZB_cons == 1:
            vort_sh = remesh(areaBu*vort_xy*sh_xy,self.h) / areaT
        
        k_bc = - amp * areaT
        S_11 = k_bc * (- vort_sh + sum_sq)
        S_22 = k_bc * (+ vort_sh + sum_sq)

        k_bc =  - amp * areaBu
        S_12 = k_bc * vort_xy * remesh(sh_xx,vort_xy)

        def ftr(x):
            return filter_iteration(x,Stress_iter,Stress_order,self.h)
        
        return self.divergence(ftr(S_11), ftr(S_12), ftr(S_22))

    def ZB_offline_cartesian(self,amplitude=1./24):
        D = self.sh_xy
        RV = self.relative_vorticity
        D_hat = self.sh_xx

        trace = 0.5 * (remesh(RV**2+D**2,self.h)+D_hat**2)

        RVD = remesh(RV*D,self.h)
        S_11 = - RVD + trace
        S_22 = + RVD + trace
        S_12 = RV * remesh(D_hat,RV)

        kappa = - amplitude * self.param.dxT * self.param.dyT
        S_11 = S_11 * kappa
        S_22 = S_22 * kappa
        S_12 = S_12 * remesh(kappa,S_12)

        fx = diffx_uq(S_11,self.u) / self.param.dxCu + diffy_tu(S_12,self.u) / self.param.dyCu
        fy = diffx_tv(S_12,self.v) / self.param.dxCv + diffy_vq(S_22,self.v) / self.param.dyCv

        return (fx,fy)