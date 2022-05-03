import xarray as xr
import os
import math
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
import numpy as np

class dataset:
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

    @property
    def RV_f(self):
        RV = self.prog.RV
        f = self.param.f
        return RV/f

class collection_of_experiments:
    def __init__(self, common_folder: str, exps: list[str], exps_names=None, additional_subfolder=''):
        '''
        exps - list of paths w.r.t. common folder; these
        relative paths are used as keys in __getitem__ method
        exps_names - default names for exeperiments
        additional_subfolder - if results are stored not in common_folder+exps[i],
        but in an additional subfolder
        '''
        self.common_folder = common_folder
        self.exps = exps

        if exps_names is None:
            exps_names = exps

        # Construct dictionary of datasets, where keys are given by exps
        self.ds = {}
        self.names = {}
        for i in range(len(exps)):
            folder = os.path.join(common_folder,exps[i],additional_subfolder)
            self.ds[exps[i]] = dataset(folder)
            self.names[exps[i]] = exps_names[i] # convert array to dictionary

    def __getitem__(self, q):
        try:
            return self.ds[q]
        except:
            print('item not found')

    ######################### service plotting functions #################
    def get_axes(self, nfig, ncol=3, size=4, ratio=1.15):
        if nfig > ncol:
            xfig=ncol
            yfig=math.ceil(nfig/ncol)
        else:
            xfig = nfig
            yfig = 1
        
        figsize_x = size * ratio * xfig
        figsize_y = size * yfig
        fig, ax = plt.subplots(yfig, xfig, figsize=(figsize_x,figsize_y), constrained_layout=True)
        try:
            ax = ax.flat # 1d array of subplots
        except:
            ax = np.array([ax,]) # to make work only one picture
        return fig, ax

    def animate(self, plot_function, Time=range(-50,0), videoname='my_movie.mp4'):
        '''
        Decorator for animation
        '''
        def new_plot_function(*args, **kwargs):
            fig, ax = self.get_axes(nfig=3,ncol=3)
            p=[]
            N = len(Time)
            use_colorbar=True
            for j in Time:
                try:
                    p_ = plot_function(*args, **kwargs, Time=j, ax=ax, use_colorbar=use_colorbar)
                except:
                    p_ = plot_function(*args, **kwargs, Time=j, ax=ax)
                print(f"{j+1} Images of {N} are plotted\r",end="")
                use_colorbar=False
                p.append([*p_])
            plt.close()

            ani = animation.ArtistAnimation(fig, p, interval=100, blit=True, repeat_delay=0)
            ani.save(videoname)
            return ani
        return new_plot_function

    #########################  snapshot plotters #########################

    def plot_relative_vorticity_snapshot(self, exps, Time=-1, zl=0, names=None, ax=None, use_colorbar=True):
        '''
        Optionally takes axes, and allows to construct facet figure. 
        Returns list of artists (p), and allows to build movies
        '''
        plt.rcParams.update({'font.size': 14})
        nfig = len(exps)
        if ax is None:
            fig, ax = self.get_axes(nfig)

        if names is None:
            names = [self.names[exp] for exp in exps]
        p = []
        for ifig, exp in enumerate(exps):
            p.append(self[exp].RV_f.isel(zl=zl,Time=Time).plot.pcolormesh(vmin=-0.2,vmax=0.2,
                ax=ax[ifig],add_colorbar=False,cmap='bwr'))
            ax[ifig].set_title(names[ifig])
        if use_colorbar:
            plt.colorbar(p[0],ax=ax,label='relative vorticity / local Coriolis ($\zeta/f$)')
        return p

    def plot_potential_vorticity_snapshot(self, exps, Time=-1, zl=0, names=None, ax=None, use_colorbar=True):
        '''
        Optionally takes axes, and allows to construct facet figure. 
        Returns list of artists (p), and allows to build movies
        '''
        plt.rcParams.update({'font.size': 14})
        nfig = len(exps)
        if ax is None:
            fig, ax = self.get_axes(nfig)

        if names is None:
            names = [self.names[exp] for exp in exps]
        p = []
        for ifig, exp in enumerate(exps):
            p.append(self[exp].prog.PV.isel(zl=zl,Time=Time).plot.pcolormesh(vmin=0,vmax=2e-7,
                ax=ax[ifig],add_colorbar=False,cmap='seismic'))
            ax[ifig].set_title(names[ifig])
        if use_colorbar:
            plt.colorbar(p[0],ax=ax,label='Potential vorticity, $m^{-1} s^{-1}$')
        return p
    
    def plot_KE_snapshot(self, exps, Time=-1, zl=0, names=None, vmax=0.05, ax=None, use_colorbar=True):
        '''
        Optionally takes axes, and allows to construct facet figure. 
        Returns list of artists (p), and allows to build movies
        '''
        plt.rcParams.update({'font.size': 14})
        nfig = len(exps)
        if ax is None:
            fig, ax = self.get_axes(nfig)

        if names is None:
            names = [self.names[exp] for exp in exps]

        p= []
        for ifig, exp in enumerate(exps):
            p.append(self[exp].energy.KE.isel(zl=zl,Time=Time).plot.pcolormesh(vmin=0,vmax=vmax,
                ax=ax[ifig],add_colorbar=False,cmap='inferno'))
            ax[ifig].set_title(names[ifig])
        if use_colorbar:
            plt.colorbar(p[0],ax=ax,label='$m^2/s^2$')
        return p

    def plot_SGS_snapshot(self, exp, Time = -1):
        fig = plt.figure(figsize=(15,7.5))
        plt.rcParams.update({'font.size': 13})
        
        mom = self[exp].mom
        fx = mom.diffu.isel(Time=Time)
        fy = mom.diffv.isel(Time=Time)
        ZBx = mom.ZB2020u.isel(Time=Time)
        ZBy = mom.ZB2020v.isel(Time=Time)
        smagx = fx - ZBx
        smagy = fy - ZBy
        x = mom.xh
        y = mom.yh
        extent = [x.min(),x.max(),y.min(),y.max()]

        fig, ax = plt.subplots(2, 4, figsize=(20,8))
        ax = ax.reshape(-1)

        def plotter(xarray, zl, num, title):
            p = ax[num].imshow(xarray.isel(zl=zl), origin='lower', 
            extent=extent, cmap=matplotlib.cm.seismic, 
            vmin=-1e-7, vmax=1e-7)
            if (num>3):
                ax[num].set_xlabel('Longitude')
            ax[num].set_title(title, fontsize=20)
            ax[num].set_xticks([0,5,10,15,20])
            ax[num].set_yticks([30,35,40,45,50])
            return p

        plotter(smagx, zl=0, num=0, title='$du/dt$, bilap Smag')
        plotter(smagy, zl=0, num=1, title='$dv/dt$, bilap Smag')
        plotter(ZBx, zl=0, num=2, title='$du/dt$, ZB2020')
        plotter(ZBy, zl=0, num=3, title='$dv/dt$, ZB2020')

        plotter(smagx, zl=1, num=4, title='$du/dt$, bilap Smag')
        plotter(smagy, zl=1, num=5, title='$dv/dt$, bilap Smag')
        plotter(ZBx, zl=1, num=6, title='$du/dt$, ZB2020')
        p = plotter(ZBy, zl=1, num=7, title='$dv/dt$, ZB2020')

        ax[0].set_ylabel('Upper layer', fontsize=20)
        ax[4].set_ylabel('Lower layer', fontsize=20)

        cbar = fig.colorbar(p, ax=ax, label='$m/s^2$')