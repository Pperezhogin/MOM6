import xarray as xr
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import numpy.fft as npfft
import dask
import math
from matplotlib import animation

def calc_ispec(kk, ll, wv, _var_dens, averaging = True, truncate=True, nd_wavenumber=False, nfactor = 1):
    """Compute isotropic spectrum `phr` from 2D spectrum of variable signal2d.

    Parameters
    ----------
    var_dens : squared modulus of fourier coefficients like this:
        np.abs(signal2d_fft)**2/m.M**2

    averaging: If True, spectral density is estimated with averaging over circles,
        otherwise summation is used and Parseval identity holds

    truncate: If True, maximum wavenumber corresponds to inner circle in Fourier space,
        otherwise - outer circle
    
    nd_wavenumber: If True, wavenumber is nondimensional: 
        minimum wavenumber is 1 and corresponds to domain length/width,
        otherwise - wavenumber is dimensional [m^-1]

    nfactor: width of the bin in sqrt(dk^2+dl^2) units

    Returns
    -------
    kr : array
        isotropic wavenumber
    phr : array
        isotropic spectrum

    Normalization:
    signal2d.var()/2 = phr.sum() * (kr[1] - kr[0])
    """

    # account for complex conjugate
    var_dens = np.copy(_var_dens)
    var_dens[...,0] /= 2
    var_dens[...,-1] /= 2

    ll_max = np.abs(ll).max()
    kk_max = np.abs(kk).max()

    dk = kk[1] - kk[0]
    dl = ll[1] - ll[0]

    if truncate:
        kmax = np.minimum(ll_max, kk_max)
    else:
        kmax = np.sqrt(ll_max**2 + kk_max**2)
    
    kmin = np.minimum(dk, dl)

    dkr = np.sqrt(dk**2 + dl**2) * nfactor

    # left border of bins
    kr = np.arange(kmin, kmax-dkr, dkr)
    
    phr = np.zeros(kr.size)

    for i in range(kr.size):
        if averaging:
            fkr =  (wv>=kr[i]) & (wv<=kr[i]+dkr)    
            if fkr.sum() == 0:
                phr[i] = 0.
            else:
                phr[i] = var_dens[fkr].mean() * (kr[i]+dkr/2) * np.pi / (dk * dl)
        else:
            fkr =  (wv>=kr[i]) & (wv<kr[i]+dkr)
            phr[i] = var_dens[fkr].sum() / dkr
    
    # convert left border of the bin to center
    kr = kr + dkr/2

    # convert to non-dimensional wavenumber 
    # preserving integral over spectrum
    if nd_wavenumber:
        kr = kr / kmin
        phr = phr * kmin

    return kr, phr

def compute_spectrum(_u, window, dx=1., dy=1., **kw):
    '''
    Input: np.array of size Ntimes * Nz * Ny * Nx,
    average dx,dy over domain (in meters;
    for spherical geometry average is given over center cells)
    '''

    # If NaNs (occasionally) are present (some boundary is defined as such,
    # but not necessary it is defined with NaNs), they are changed to 0
    u = np.nan_to_num(_u)

    nx = u.shape[-1]
    ny = u.shape[-2]

    # subtract spatial mean (as our spectra ignore this characteristic,
    # and there may be spectral leakage in a case of window)
    u = u - u.mean(axis=(-1,-2), keepdims=True)

    # apply window
    if window == 'rect':
        Wnd = np.ones((ny,nx))
    elif window == 'hanning':
        Wnd = np.outer(np.hanning(ny),np.hanning(nx))
    elif window == 'hamming':
        Wnd = np.outer(np.hamming(ny),np.hamming(nx))
    elif window == 'bartlett':
        Wnd = np.outer(np.bartlett(ny),np.bartlett(nx))
    elif window == 'blackman':
        Wnd = np.outer(np.blackman(ny),np.blackman(nx))
    elif window == 'kaiser':
        Wnd = np.outer(np.kaiser(ny,14),np.kaiser(nx,14))
    else:
        print('wrong window')

    # compensation of Parseval identity
    Wnd_sqr = (Wnd**2).mean()

    # Pointwise multiplication occurs only at last two dimensions
    # see https://numpy.org/doc/stable/reference/generated/numpy.multiply.html
    u = u * Wnd

    uf = npfft.rfftn(u, axes=(-2,-1))
    M = u.shape[-1] * u.shape[-2] # total number of points

    u2 = (np.abs(uf)**2 / M**2) / Wnd_sqr

    if len(u2.shape) == 3:
        u2 = u2.mean(axis=0)
    elif len(u2.shape) > 3:
        print('error')

    # maximum wavenumber is 1/(2*d) = pi/dx = pi/dy
    kx = npfft.rfftfreq(nx,d=dx/(2*np.pi))
    ky = npfft.fftfreq(ny,d=dy/(2*np.pi))

    Kx, Ky = np.meshgrid(kx, ky)
    K = np.sqrt(Kx**2+Ky**2)
    
    return calc_ispec(kx, ky, K, u2, **kw)

def compute_cospectrum(_u, _f, window, dx=1., dy=1., **kw):
    u = np.nan_to_num(_u)
    f = np.nan_to_num(_f)

    nx = u.shape[-1]
    ny = u.shape[-2]

    u = u - u.mean(axis=(-1,-2), keepdims=True)
    f = f - f.mean(axis=(-1,-2), keepdims=True)

    if window == 'rect':
        Wnd = np.ones((ny,nx))
    elif window == 'hanning':
        Wnd = np.outer(np.hanning(ny),np.hanning(nx))
    elif window == 'hamming':
        Wnd = np.outer(np.hamming(ny),np.hamming(nx))
    elif window == 'bartlett':
        Wnd = np.outer(np.bartlett(ny),np.bartlett(nx))
    elif window == 'blackman':
        Wnd = np.outer(np.blackman(ny),np.blackman(nx))
    elif window == 'kaiser':
        Wnd = np.outer(np.kaiser(ny,14),np.kaiser(nx,14))
    else:
        print('wrong window')

    Wnd_sqr = (Wnd**2).mean()

    u = u * Wnd
    f = f * Wnd

    uf = npfft.rfftn(u, axes=(-2,-1))
    ff = npfft.rfftn(f, axes=(-2,-1))
    M = u.shape[-1] * u.shape[-2] # total number of points

    cosp = np.real(uf * np.conj(ff) / M**2) / Wnd_sqr

    if len(cosp.shape) == 3:
        cosp = cosp.mean(axis=0)
    elif len(cosp.shape) > 3:
        print('error')

    # maximum wavenumber is 1/(2*d) = pi/dx = pi/dy
    kx = npfft.rfftfreq(nx,d=dx/(2*np.pi))
    ky = npfft.fftfreq(ny,d=dy/(2*np.pi))

    Kx, Ky = np.meshgrid(kx, ky)
    K = np.sqrt(Kx**2+Ky**2)

    return calc_ispec(kx, ky, K, 2*cosp, **kw)

def compute_cospectrum_uv(u, v, fx, fy, window, dx, dy, **kw):
    kx, Ekx = compute_cospectrum(u, fx, window, dx, dy, **kw)
    ky, Eky = compute_cospectrum(v, fy, window, dx, dy, **kw)
    return kx, Ekx+Eky

class dataset:
    def __init__(self, folder):
        self.series = xr.open_dataset(folder+'/ocean.stats.nc', decode_times=False)
        self.ave = xr.open_mfdataset(folder+'/ave_*.nc', decode_times=False)
        self.prog = xr.open_mfdataset(folder+'/prog_*.nc', decode_times=False)
        self.param = xr.open_dataset(folder+'/ocean_geometry.nc', decode_times=False)
        self.energy = xr.open_mfdataset(folder+'/energy_*.nc', decode_times=False)
        self.forcing = xr.open_mfdataset(folder+'/forcing_*.nc', decode_times=False)
        self.mom = xr.open_mfdataset(folder+'/mom_*.nc', decode_times=False)

class dataset_experiments:
    def __init__(self, common_folder, exps, exps_names=None):
        self.common_folder = common_folder
        self.exps = exps

        if exps_names is None:
            self.exps_names = exps
        else:
            self.exps_names = exps_names

        self.ds = {}
        self.names = {}
        for i in range(len(exps)):
            folder = common_folder + '/' + exps[i] + '/output'
            self.ds[exps[i]] = dataset(folder)
            self.names[exps[i]] = self.exps_names[i] # convert array to dictionary

    def __getitem__(self, q):
        try:
            return self.ds[q]
        except:
            print('item not found')

    def plot_domain(self, exp, tstart=3650.):
        def plot(axes, xangle, yangle):
            ave = self[exp].ave
            forcing = self[exp].forcing

            topography = ave.e.isel(zi=2)[ave.Time>tstart].mean(dim='Time')
            free_surface = ave.e.isel(zi=0)[ave.Time>tstart].mean(dim='Time')
            interface = ave.e.isel(zi=1)[ave.Time>tstart].mean(dim='Time')
            h = ave.h.isel(zl=1)[ave.Time>tstart].mean(dim='Time')
            taux = forcing.taux.isel(Time=-1,xq=1)

            mask_interface = np.ones_like(h)
            mask_interface[h<0.0001] = np.nan

            xh = ave.xh
            yh = ave.yh
            X, Y = np.meshgrid(xh, yh)

            plt.rcParams.update({'font.size': 12})
            p1 = axes.plot_surface(X,Y,topography, label='topography', edgecolor='none', alpha=0.3)
            p2 = axes.plot_surface(X,Y,interface * mask_interface, label='interface', edgecolor='none', alpha=0.7)
            p3 = axes.plot_surface(X,Y,free_surface, label='free surface', edgecolor='none', alpha=0.3)

            import pdb
            #pdb.set_trace()

            yy = yh
            xx = np.ones_like(yy) * float(xh.min())
            zz = np.ones_like(yy) * 100

            skip = slice(None, None, 25)

            axes.quiver(xx[skip], yy[skip], zz[skip], taux[skip], taux[skip]*0, taux[skip]*0, length = 100, alpha=1.0, linewidth=2)

            axes.contour3D(X, Y, free_surface, levels=np.arange(-4,4,0.5),colors='k')

            axes.view_init(xangle, yangle)

            # https://stackoverflow.com/questions/55531760/is-there-a-way-to-label-multiple-3d-surfaces-in-matplotlib/55534939
            
            p1._facecolors2d = p1._facecolor3d
            p1._edgecolors2d = p1._facecolor3d

            p2._facecolors2d = p2._facecolor3d
            p2._edgecolors2d = p2._facecolor3d

            p3._facecolors2d = p3._facecolor3d
            p3._edgecolors2d = p3._facecolor3d
             
            axes.set_xlabel('Longitude')
            axes.set_ylabel('Latitude')
            axes.set_zlabel('depth, $m$')
            axes.set_yticks([30,35,40,45,50])
            axes.set_zticks([0, -500, -1000, -1500, -2000])
            axes.legend()

        fig = plt.figure(figsize=(15,5), tight_layout = True)
        axes = fig.add_subplot(1, 3, 1, projection='3d')
        plot(axes, 50, 200)
        axes = fig.add_subplot(1, 3, 2, projection='3d')
        plot(axes, 20, 200)
        axes = fig.add_subplot(1, 3, 3, projection='3d')
        plot(axes, 20, 240)

    def plot_KE(self, exps, tstart=3650., names=None):
        fig = plt.figure(figsize=(13,5))
        plt.rcParams.update({'font.size': 18})

        if names is None:
            names = []
            for exp in exps:
                names.append(self.names[exp])

        for iexp,exp in enumerate(exps):
            plt.subplot(121)
            series = self[exp].series
            t = series.Time
            KE = series.KE.isel(Layer=0) / series.Mass
            KE_mean = KE[t >= tstart].mean()
            p = plt.plot(t/365, KE, label=names[iexp])
            color = p[0].get_color()
            plt.axhline(y = KE_mean, linestyle='--', color=color)
            plt.xlabel('Time, years')
            plt.ylabel('$m^2/s^2$')
            plt.title('Kinetic Energy, upper layer')   
            plt.legend()
            
            plt.subplot(122)
            series = self[exp].series
            t = series.Time
            KE = series.KE.isel(Layer=1) / series.Mass
            KE_mean = KE[t >= tstart].mean()
            p = plt.plot(t/365, KE, label=names[iexp])
            color = p[0].get_color()
            plt.axhline(y = KE_mean, linestyle='--', color=color)
            plt.xlabel('Time, years')
            plt.ylabel('$m^2/s^2$')
            plt.title('Kinetic Energy, lower layer') 
        plt.tight_layout()

    def plot_ssh(self, exps, tstart=3650., names=None):
        nfig = len(exps)
        plt.rcParams.update({'font.size': 12})

        if names is None:
            names = []
            for exp in exps:
                names.append(self.names[exp])

        if nfig > 3:
            xfig=3
            yfig=math.ceil(nfig/3)
        else:
            xfig = nfig
            yfig = 1

        fig = plt.figure(figsize=(xfig*4,yfig*4))

        for ifig, exp in enumerate(exps):
            #plt.subplot(int(str(yfig)+str(xfig)+str(ifig+1)))
            plt.subplot(yfig,xfig,ifig+1)
            ave = self[exp].ave
            t = ave.Time
            ssh = ave.e.isel(zi=0)[t >= tstart].mean(dim='Time')
            xh = ave.xh
            yh = ave.yh
            X, Y = np.meshgrid(xh,yh)
            ax = plt.gca()
            Cplot = plt.contour(X,Y,ssh, levels=np.arange(-4,4,0.5), colors='k', linewidths=1)
            ax.clabel(Cplot, Cplot.levels)
            plt.xticks((0, 5, 10, 15, 20))
            plt.yticks((30, 35, 40, 45, 50))
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title(names[ifig])

        plt.tight_layout()

    def plot_relative_vorticity_snapshot(self, exps, Time=-1, zl=0, names=None):
        nfig = len(exps)
        plt.rcParams.update({'font.size': 12})

        if names is None:
            names = []
            for exp in exps:
                names.append(self.names[exp])

        if nfig > 3:
            xfig=3
            yfig=math.ceil(nfig/3)
        else:
            xfig = nfig
            yfig = 1

        fig, ax = plt.subplots(yfig, xfig, figsize=(xfig*5,yfig*4))
        ax = ax.reshape(-1)
        for ifig, exp in enumerate(exps):
            prog = self[exp].prog
            RV = np.array(prog.RV.isel(zl=zl, Time=Time))
            param = self[exp].param
            f = np.array(param.f)
            xq = prog.xq
            yq = prog.yq
            p = ax[ifig].imshow(RV / f, origin='lower',
                extent=[xq.min(),xq.max(),yq.min(),yq.max()], 
                cmap='bwr', vmin=-0.2, vmax = 0.2)
            ax[ifig].set_xlabel('Longitude')
            ax[ifig].set_title(names[ifig])

        ax[0].set_ylabel('Latitude')
        if (yfig>1):
            ax[xfig].set_ylabel('Latitude')
        
        cbar = fig.colorbar(p, ax=ax, label='relative vorticity / local Coriolis ($\zeta/f$)')

    def plot_relative_vorticity_animation(self, exps, timeover=50, Time=-1, zl=0):
        nfig = len(exps)
        plt.rcParams.update({'font.size': 12})

        if nfig > 3:
            xfig = int(nfig / 2)
            yfig = 2
        else:
            xfig = nfig
            yfig = 1

        fig, ax = plt.subplots(yfig, xfig, figsize=(xfig*5,yfig*4), frameon=False)
        ax = ax.reshape(-1)
        frames = [] # store generated images
        for i in range(timeover):
            for ifig, exp in enumerate(exps):
                prog = self[exp].prog
                RV = np.array(prog.RV.isel(zl=zl, Time=Time-timeover+i))
                param = self[exp].param
                f = np.array(param.f)
                xq = prog.xq
                yq = prog.yq
                globals()['p'+str(ifig)] = ax[ifig].imshow(RV / f, origin='lower',
                    extent=[xq.min(),xq.max(),yq.min(),yq.max()], 
                    cmap='bwr', vmin=-0.2, vmax = 0.2)
                ax[ifig].set_xlabel('Longitude')
                ax[ifig].set_title(self.names[exp])

            if i==1:
                ax[0].set_ylabel('Latitude')
                if (yfig>1):
                    ax[xfig].set_ylabel('Latitude')
                
                fig.colorbar(globals()['p'+str(nfig-1)], ax=ax, label='N/D units')
            frames.append([globals()['p'+str(ifig)] for ifig in range(nfig)])
        ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True,
                                repeat_delay=1000)

        plt.close(fig)
        video_name = 'vorti_anim_'+self.common_folder.split('/')[-1]+'.mp4'
        ani.save(video_name)
        
        return video_name, fig

    def plot_KE_animation(self, exps, timeover=50, Time=-1, zl=0, vmax=0.05):
        nfig = len(exps)
        plt.rcParams.update({'font.size': 12})

        if nfig > 3:
            xfig = int(nfig / 2)
            yfig = 2
        else:
            xfig = nfig
            yfig = 1

        fig, ax = plt.subplots(yfig, xfig, figsize=(xfig*5,yfig*4), frameon=False)
        ax = ax.reshape(-1)
        frames = [] # store generated images
        for i in range(timeover):
            for ifig, exp in enumerate(exps):
                energy = self[exp].energy
                KE = np.array(energy.KE.isel(zl=zl, Time=Time-timeover+i))
                xh = energy.xh
                yh = energy.yh
                globals()['p'+str(ifig)] = ax[ifig].imshow(KE, origin='lower',
                    extent=[xh.min(),xh.max(),yh.min(),yh.max()], 
                    cmap='inferno', vmin=0, vmax = vmax)
                ax[ifig].set_xlabel('Longitude')
                ax[ifig].set_title(self.names[exp])

            if i==1:
                ax[0].set_ylabel('Latitude')
                if (yfig>1):
                    ax[xfig].set_ylabel('Latitude')
                
                fig.colorbar(globals()['p'+str(nfig-1)], ax=ax, label='$m^2/s^2$')
            frames.append([globals()['p'+str(ifig)] for ifig in range(nfig)])
        ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True,
                                repeat_delay=1000)

        plt.close(fig)
        video_name = 'vorti_anim_'+self.common_folder.split('/')[-1]+'.mp4'
        ani.save(video_name)
        
        return video_name, fig

    def plot_KE_snapshot(self, exps, Time=-1, zl=0):
        nfig = len(exps)
        plt.rcParams.update({'font.size': 12})

        if nfig > 3:
            xfig=3
            yfig=math.ceil(nfig/3)
        else:
            xfig = nfig
            yfig = 1

        fig, ax = plt.subplots(yfig, xfig, figsize=(xfig*5,yfig*4))
        ax = ax.reshape(-1)
        for ifig, exp in enumerate(exps):
            energy = self[exp].energy
            KE = np.array(energy.KE.isel(zl=zl, Time=Time))
            xh = energy.xh
            yh = energy.yh
            p = ax[ifig].imshow(KE, origin='lower',
                extent=[xh.min(),xh.max(),yh.min(),yh.max()], 
                cmap='inferno', vmin=0, vmax=0.05)
            ax[ifig].set_xlabel('Longitude')
            ax[ifig].set_title(self.names[exp])

        ax[0].set_ylabel('Latitude')
        if (yfig>1):
            ax[xfig].set_ylabel('Latitude')
        
        fig.colorbar(p, ax=ax, label='$m^2/s^2$')

    def plot_EKE(self, exps, tstart=3650., zl=0, vmax = 0.02, names=None):
        nfig = len(exps)
        plt.rcParams.update({'font.size': 12})

        if names is None:
            names = []
            for exp in exps:
                names.append(self.names[exp])

        if nfig > 3:
            xfig=3
            yfig=math.ceil(nfig/3)
        else:
            xfig = nfig
            yfig = 1

        fig, ax = plt.subplots(yfig, xfig, figsize=(xfig*5,yfig*4))
        ax = ax.reshape(-1)
        for ifig, exp in enumerate(exps):
            energy = self[exp].energy
            ave = self[exp].ave
            KE_full = np.array(energy.KE.isel(zl=zl)[energy.Time>=tstart].mean(dim='Time'))
            u = ave.u.isel(zl=zl)[ave.Time>=tstart].mean(dim='Time').data
            v = ave.v.isel(zl=zl)[ave.Time>=tstart].mean(dim='Time').data
            u2 = np.array(u)**2
            v2 = np.array(v)**2
            KE_mean = 0.25 * (u2[:,1:] + u2[:,0:-1] + v2[1:,:] + v2[0:-1,:])
            
            EKE = KE_full-KE_mean

            xh = energy.xh
            yh = energy.yh
            X,Y = np.meshgrid(xh, yh)
            p = ax[ifig].imshow(EKE, origin='lower',
                extent=[xh.min(),xh.max(),yh.min(),yh.max()], 
                cmap='inferno', vmin = 0, vmax = vmax)
            ax[ifig].set_xlabel('Longitude')
            ax[ifig].set_title(names[ifig])

        ax[0].set_ylabel('Latitude')
        if (yfig>1):
            ax[xfig].set_ylabel('Latitude')
        
        fig.colorbar(p, ax=ax, label='$m^2/s^2$')

    def plot_KE_spectrum(self, exps, names=None, tstart=7200., Lat=(30,50), Lon=(0,22), window='rect', **kw):
        if names is None:
            names = []
            for exp in exps:
                names.append(self.names[exp])
        
        fig = plt.figure(figsize=(13,5))
        plt.rcParams.update({'font.size': 16})
        for jfig, exp in enumerate(exps):
            prog = self[exp].prog
            param = self[exp].param
            t = prog.Time
            xq = prog.xq
            yq = prog.yq
            xh = prog.xh
            yh = prog.yh
            dxT = param.dxT
            dyT = param.dyT
            lonh = param.lonh
            lath = param.lath
            with dask.config.set(**{'array.slicing.split_large_chunks': False}):
                u = np.array(prog.u[t>=tstart,:,(yh>Lat[0])*(yh<=Lat[1]),(xq>Lon[0])*(xq<=Lon[1])])
                v = np.array(prog.v[t>=tstart,:,(yq>Lat[0])*(yq<=Lat[1]),(xh>Lon[0])*(xh<=Lon[1])])
                dx = float(dxT[(lath>Lat[0])*(lath<=Lat[1]),(lonh>Lon[0])*(lonh<=Lon[1])].mean())
                dy = float(dyT[(lath>Lat[0])*(lath<=Lat[1]),(lonh>Lon[0])*(lonh<=Lon[1])].mean())
            
            plt.subplot(121)
            k, Eu = compute_spectrum(u[:,0,:,:], window, dx, dy, **kw)
            k, Ev = compute_spectrum(v[:,0,:,:], window, dx, dy, **kw)
            plt.loglog(k,Eu+Ev, label=names[jfig])
            plt.xlabel(r'wavenumber, $k [m^{-1}]$')
            plt.ylabel(r'Energy spectrum, $E(k) [m^3/s^2]$')
            plt.title('Upper layer')
            plt.legend(prop={'size': 14})

            plt.subplot(122)
            k, Eu = compute_spectrum(u[:,1,:,:], window, dx, dy, **kw)
            k, Ev = compute_spectrum(v[:,1,:,:], window, dx, dy, **kw)
            plt.loglog(k,Eu+Ev, label=names[jfig])
            plt.xlabel(r'wavenumber, $k [m^{-1}]$')
            plt.ylabel(r'Energy spectrum, $E(k) [m^3/s^2]$')
            plt.title('Lower layer')
        
            '''
            plt.subplot(121)
            k = [20, 300]
            E = [2e-4, 0]
            E[1] = E[0] * (k[1]/k[0])**(-3)
            plt.loglog(k,E,'--k')
            plt.text(100, 1e-5, '$k^{-3}$')

            plt.subplot(122)
            k = [70, 300]
            E = [2e-6, 0]
            E[1] = E[0] * (k[1]/k[0])**(-3)
            plt.loglog(k,E,'--k')
            plt.text(100, 2e-6, '$k^{-3}$')
            '''

        plt.subplot(121)
        plt.grid(which='both',linestyle=':')
        k = [5e-5, 1e-3]
        E = [1.5e+2, 0]
        E[1] = E[0] * (k[1]/k[0])**(-3)
        plt.loglog(k,E,'--k')
        plt.text(2e-4,1e+1,'$k^{-3}$')
        plt.xlim([5e-6, 2e-3])
        plt.subplot(122)
        plt.grid(which='both',linestyle=':')
        plt.xlim([5e-6, 2e-3])
        k = [5e-5, 1e-3]
        E = [3e+1, 0]
        E[1] = E[0] * (k[1]/k[0])**(-3)
        plt.loglog(k,E,'--k')
        plt.text(2e-4,1e+1,'$k^{-3}$')
        plt.tight_layout()

    def plot_cospectrum(self, exps, tstart = 7200., Lat=(30,50), Lon=(0,22), window='rect', **kw):
        fig = plt.figure(figsize=(13,5))
        plt.rcParams.update({'font.size': 16})
        for exp in exps:
            prog = self[exp].prog
            mom = self[exp].mom
            param = self[exp].param
            t = prog.Time
            xq = prog.xq
            yq = prog.yq
            xh = prog.xh
            yh = prog.yh
            dxT = param.dxT
            dyT = param.dyT
            lonh = param.lonh
            lath = param.lath

            with dask.config.set(**{'array.slicing.split_large_chunks': False}):
                u = np.array(prog.u[t>=tstart,:,(yh>Lat[0])*(yh<=Lat[1]),(xq>Lon[0])*(xq<=Lon[1])])
                v = np.array(prog.v[t>=tstart,:,(yq>Lat[0])*(yq<=Lat[1]),(xh>Lon[0])*(xh<=Lon[1])])
                fx = np.array(mom.diffu[t>=tstart,:,(yh>Lat[0])*(yh<=Lat[1]),(xq>Lon[0])*(xq<=Lon[1])])
                fy = np.array(mom.diffv[t>=tstart,:,(yq>Lat[0])*(yq<=Lat[1]),(xh>Lon[0])*(xh<=Lon[1])])
                dx = float(dxT[(lath>Lat[0])*(lath<=Lat[1]),(lonh>Lon[0])*(lonh<=Lon[1])].mean())
                dy = float(dyT[(lath>Lat[0])*(lath<=Lat[1]),(lonh>Lon[0])*(lonh<=Lon[1])].mean())

            plt.subplot(121)
            k, E = compute_cospectrum_uv(u[:,0,:,:], v[:,0,:,:], fx[:,0,:,:], fy[:,0,:,:], window, dx, dy, **kw)
            plt.semilogx(k,E*k, label=self.names[exp])
            plt.xlabel('$k$, wavenumber')
            plt.ylabel(r'$k \oint Re(\mathbf{u}_k \mathbf{f}_k^*) dk$')
            plt.title('Upper layer')
            plt.ylim([-7e-10,0.5e-10])

            plt.subplot(122)
            k, E = compute_cospectrum_uv(u[:,1,:,:], v[:,1,:,:], fx[:,1,:,:], fy[:,1,:,:], window, dx, dy, **kw)
            plt.semilogx(k,E*k, label=self.names[exp])
            plt.xlabel('$k$, wavenumber')
            plt.title('Lower layer')
            plt.ylim([-1.2e-10,0.1e-10])
            plt.legend()
            
        plt.tight_layout()

    def plot_cospectrum_componentwise(self, exps, tstart = 7200., Lat=(30,50), Lon=(0,22), window='rect', ylim1=(-1.5e-9,1.5e-9), ylim2=(-2e-10,2e-10), **kw):
        fig = plt.figure(figsize=(13,4.5))
        plt.rcParams.update({'font.size': 16})
        for exp in exps:
            prog = self[exp].prog
            mom = self[exp].mom
            param = self[exp].param
            t = prog.Time
            xq = prog.xq
            yq = prog.yq
            xh = prog.xh
            yh = prog.yh
            dxT = param.dxT
            dyT = param.dyT
            lonh = param.lonh
            lath = param.lath

            with dask.config.set(**{'array.slicing.split_large_chunks': False}):
                u = np.array(prog.u[t>=tstart,:,(yh>Lat[0])*(yh<=Lat[1]),(xq>Lon[0])*(xq<=Lon[1])])
                v = np.array(prog.v[t>=tstart,:,(yq>Lat[0])*(yq<=Lat[1]),(xh>Lon[0])*(xh<=Lon[1])])
                fx = np.array(mom.diffu[t>=tstart,:,(yh>Lat[0])*(yh<=Lat[1]),(xq>Lon[0])*(xq<=Lon[1])])
                fy = np.array(mom.diffv[t>=tstart,:,(yq>Lat[0])*(yq<=Lat[1]),(xh>Lon[0])*(xh<=Lon[1])])
                ZBx = np.array(mom.ZB2020u[t>=tstart,:,(yh>Lat[0])*(yh<=Lat[1]),(xq>Lon[0])*(xq<=Lon[1])])
                ZBy = np.array(mom.ZB2020v[t>=tstart,:,(yq>Lat[0])*(yq<=Lat[1]),(xh>Lon[0])*(xh<=Lon[1])])
                dx = float(dxT[(lath>Lat[0])*(lath<=Lat[1]),(lonh>Lon[0])*(lonh<=Lon[1])].mean())
                dy = float(dyT[(lath>Lat[0])*(lath<=Lat[1]),(lonh>Lon[0])*(lonh<=Lon[1])].mean())

            plt.subplot(121)
            k, E = compute_cospectrum_uv(u[:,0,:,:], v[:,0,:,:], fx[:,0,:,:], fy[:,0,:,:], window, dx, dy, **kw)
            k, EZB = compute_cospectrum_uv(u[:,0,:,:], v[:,0,:,:], ZBx[:,0,:,:], ZBy[:,0,:,:], window, dx, dy, **kw)
            Esmag = E - EZB
            plt.semilogx(k,E*k, label='sum')
            plt.semilogx(k,EZB*k, '--', label='ZB2020')
            plt.semilogx(k,Esmag*k, '-.', label='bilap Smag')
            plt.axhline(y=0,color='k', linestyle='--', alpha=0.5)
            plt.xlabel(r'wavenumber, $k [m^{-1}]$')
            plt.ylabel(r'$k \oint Re(\mathbf{u}_k \mathbf{f}_k^*) dk ~[m^2/s^3]$')
            plt.title('Upper layer')
            plt.legend(fontsize=12)
            plt.ylim(ylim1)
            
            plt.subplot(122)
            k, E = compute_cospectrum_uv(u[:,1,:,:], v[:,1,:,:], fx[:,1,:,:], fy[:,1,:,:], window, dx, dy, **kw)
            k, EZB = compute_cospectrum_uv(u[:,1,:,:], v[:,1,:,:], ZBx[:,1,:,:], ZBy[:,1,:,:], window, dx, dy, **kw)
            Esmag = E - EZB
            plt.semilogx(k,E*k, label='sum')
            plt.semilogx(k,EZB*k, '--', label='ZB2020')
            plt.semilogx(k,Esmag*k, '-.', label='bilap Smag')
            plt.axhline(y=0,color='k', linestyle='--', alpha=0.5)
            plt.xlabel(r'wavenumber, $k [m^{-1}]$')
            plt.title('Lower layer')
            plt.legend(fontsize=12)
            plt.ylim(ylim2)
            
        plt.tight_layout()

    def plot_cospectrum_spectrum(self, exps, tstart = 7200., Lat=(30,50), Lon=(0,22), window='rect', **kw):
        fig = plt.figure(figsize=(13,5))
        plt.rcParams.update({'font.size': 16})
        for exp in exps:
            prog = self[exp].prog
            mom = self[exp].mom
            param = self[exp].param
            t = prog.Time
            xq = prog.xq
            yq = prog.yq
            xh = prog.xh
            yh = prog.yh
            dxT = param.dxT
            dyT = param.dyT
            lonh = param.lonh
            lath = param.lath

            with dask.config.set(**{'array.slicing.split_large_chunks': False}):
                u = np.array(prog.u[t>=tstart,:,(yh>Lat[0])*(yh<=Lat[1]),(xq>Lon[0])*(xq<=Lon[1])])
                v = np.array(prog.v[t>=tstart,:,(yq>Lat[0])*(yq<=Lat[1]),(xh>Lon[0])*(xh<=Lon[1])])
                fx = np.array(mom.diffu[t>=tstart,:,(yh>Lat[0])*(yh<=Lat[1]),(xq>Lon[0])*(xq<=Lon[1])])
                fy = np.array(mom.diffv[t>=tstart,:,(yq>Lat[0])*(yq<=Lat[1]),(xh>Lon[0])*(xh<=Lon[1])])
                ZBx = np.array(mom.ZB2020u[t>=tstart,:,(yh>Lat[0])*(yh<=Lat[1]),(xq>Lon[0])*(xq<=Lon[1])])
                ZBy = np.array(mom.ZB2020v[t>=tstart,:,(yq>Lat[0])*(yq<=Lat[1]),(xh>Lon[0])*(xh<=Lon[1])])
                dx = float(dxT[(lath>Lat[0])*(lath<=Lat[1]),(lonh>Lon[0])*(lonh<=Lon[1])].mean())
                dy = float(dyT[(lath>Lat[0])*(lath<=Lat[1]),(lonh>Lon[0])*(lonh<=Lon[1])].mean())

            smagx = fx - ZBx
            smagy = fy - ZBy

            plt.subplot(121)
            k, E = compute_cospectrum_uv(u[:,0,:,:], v[:,0,:,:], fx[:,0,:,:], fy[:,0,:,:], window, dx, dy, **kw)
            k, EZB = compute_cospectrum_uv(u[:,0,:,:], v[:,0,:,:], ZBx[:,0,:,:], ZBy[:,0,:,:], window, dx, dy, **kw)
            Esmag = E - EZB
            plt.semilogx(k,E*k, label='sum')
            plt.semilogx(k,EZB*k, '--', label='ZB2020')
            plt.semilogx(k,Esmag*k, '-.', label='Smag')
            plt.axhline(y=0,color='k', linestyle='--', alpha=0.5)
            plt.xlabel('$k$, wavenumber')
            plt.ylabel(r'$k \oint Re(\mathbf{u}_k \mathbf{f}_k^*) dk$')
            plt.title('Energy cospectrum')
            plt.ylim((-7e-10,4e-10))
            
            plt.subplot(122)
            k, E = compute_cospectrum_uv(fx[:,0,:,:], fy[:,0,:,:], fx[:,0,:,:], fy[:,0,:,:], window, dx, dy, **kw)
            k, EZB = compute_cospectrum_uv(ZBx[:,0,:,:], ZBy[:,0,:,:], ZBx[:,0,:,:], ZBy[:,0,:,:], window, dx, dy, **kw)
            k, Esmag = compute_cospectrum_uv(smagx[:,0,:,:], smagy[:,0,:,:], smagx[:,0,:,:], smagy[:,0,:,:], window, dx, dy, **kw)
            plt.semilogx(k,E, label='sum')
            plt.semilogx(k,EZB, '--', label='ZB2020')
            plt.semilogx(k,Esmag, '-.', label='Smag')
            plt.xlabel('$k$, wavenumber')
            plt.title('Power spectrum')
            plt.legend()
            #plt.ylim((0, 1.5e-16))
            
        plt.tight_layout()


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

    def plot_energy_tendency(self, exps):
        nfig = len(exps)
        plt.rcParams.update({'font.size': 12})

        if nfig > 3:
            xfig = int(nfig / 2)
            yfig = 2
        else:
            xfig = nfig
            yfig = 1

        fig, ax = plt.subplots(yfig, xfig, figsize=(xfig*5,yfig*4))
        ax = ax.reshape(-1)
        for ifig, exp in enumerate(exps):
            energy = self[exp].energy
            param = self[exp].param
            dx = param.dxT
            dy = param.dyT
            time = energy.Time

            dKE = np.array(energy.KE_horvisc)
            dKE_ZB = np.array(energy.KE_ZB2020)
            dKE_horvisc = dKE - dKE_ZB

            Nt, Nz = dKE.shape[0], dKE.shape[1]
            
            dKE_ZB_time = np.zeros(Nt)
            dKE_horvisc_time = np.zeros(Nt)
            for nt in range(Nt):
                for nz in range(Nz):
                    dKE_ZB_time[nt] += np.sum(dKE_ZB[nt,nz,:,:] * dx * dy)
                    dKE_horvisc_time[nt] += np.sum(dKE_horvisc[nt,nz,:,:] * dx * dy)

            plt.subplot(121)
            plt.plot(time, dKE_horvisc_time)
            plt.xlabel('Time, days')
            plt.title('Energy tendency due to eddy viscosity')
            plt.ylabel('$m^5/s^3$')
            
            plt.subplot(122)
            plt.semilogy(time, np.abs(dKE_ZB_time), label=self.names[exp])
            plt.xlabel('Time, days')
            plt.title('ABS energy tendency due to ZB')
            plt.ylabel('$m^5/s^3$')
            plt.legend()