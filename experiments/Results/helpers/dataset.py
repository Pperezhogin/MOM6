import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as npfft

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
    kr = np.arange(kmin, kmax, dkr)
    
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

def compute_spectrum(u, dx=1, dy=1, **kw):
    uf = npfft.rfftn(np.nan_to_num(u), axes=(-2,-1))
    M = u.shape[-1] * u.shape[-2] # total number of points

    u2 = (np.abs(uf)**2 / M**2)

    if len(u2.shape) == 3:
        u2 = u2.mean(axis=0)
    elif len(u2.shape) > 3:
        print('error')

    Lx = u.shape[-1] * dx
    Ly = u.shape[-2] * dy

    kx_max = np.pi/dx
    kx_step = 2*np.pi/Lx
    ky_max = np.pi/dy
    ky_step = 2*np.pi/Ly

    if u.shape[-1] % 2 == 0:
        kx = np.arange(0,kx_max+kx_step,kx_step)
    else:
        kx = np.arange(0,kx_max,kx_step)
    if u.shape[-2] % 2 == 0:
        ky = np.hstack([np.arange(0,ky_max,ky_step), np.flip(np.arange(ky_step,ky_max+ky_step,ky_step))])    
    else:
        ky = np.hstack([np.arange(0,ky_max,ky_step), np.flip(np.arange(ky_step,ky_max,ky_step))])

    Kx, Ky = np.meshgrid(kx, ky)
    K = np.sqrt(Kx**2+Ky**2)

    return calc_ispec(kx, ky, K, u2, nd_wavenumber=True, **kw)

def compute_cospectrum(u, f, dx=1, dy=1, **kw):
    uf = npfft.rfftn(np.nan_to_num(u), axes=(-2,-1))
    ff = npfft.rfftn(np.nan_to_num(f), axes=(-2,-1))
    M = u.shape[-1] * u.shape[-2] # total number of points

    cosp = np.real(uf * np.conj(ff) / M**2)

    if len(cosp.shape) == 3:
        cosp = cosp.mean(axis=0)
    elif len(u2.shape) > 3:
        print('error')

    Lx = u.shape[-1] * dx
    Ly = u.shape[-2] * dy

    kx_max = np.pi/dx
    kx_step = 2*np.pi/Lx
    ky_max = np.pi/dy
    ky_step = 2*np.pi/Ly

    if u.shape[-1] % 2 == 0:
        kx = np.arange(0,kx_max+kx_step,kx_step)
    else:
        kx = np.arange(0,kx_max,kx_step)
    if u.shape[-2] % 2 == 0:
        ky = np.hstack([np.arange(0,ky_max,ky_step), np.flip(np.arange(ky_step,ky_max+ky_step,ky_step))])    
    else:
        ky = np.hstack([np.arange(0,ky_max,ky_step), np.flip(np.arange(ky_step,ky_max,ky_step))])

    Kx, Ky = np.meshgrid(kx, ky)
    K = np.sqrt(Kx**2+Ky**2)

    return calc_ispec(kx, ky, K, 2*cosp, nd_wavenumber=True, **kw)

def compute_cospectrum_uv(u, v, fx, fy, **kw):
    kx, Ekx = compute_cospectrum(u, fx, **kw)
    ky, Eky = compute_cospectrum(v, fy, **kw)
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

    def plot_KE(self, exps, tstart=3650.):
        fig = plt.figure(figsize=(13,5))
        plt.rcParams.update({'font.size': 18})
        for exp in exps:
            plt.subplot(121)
            series = self[exp].series
            t = series.Time
            KE = series.KE.isel(Layer=0) / series.Mass
            KE_mean = KE[t >= tstart].mean()
            p = plt.plot(t/365, KE, label=self.names[exp])
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
            p = plt.plot(t/365, KE, label=self.names[exp])
            color = p[0].get_color()
            plt.axhline(y = KE_mean, linestyle='--', color=color)
            plt.xlabel('Time, years')
            plt.ylabel('$m^2/s^2$')
            plt.title('Kinetic Energy, lower layer') 
        plt.tight_layout()

    def plot_ssh(self, exps, tstart=3650.):
        nfig = len(exps)
        plt.rcParams.update({'font.size': 16})

        if nfig > 3:
            xfig = int(nfig / 2)
            yfig = 2
        else:
            xfig = nfig
            yfig = 1

        fig = plt.figure(figsize=(xfig*4,yfig*4))

        for ifig, exp in enumerate(exps):
            plt.subplot(int(str(yfig)+str(xfig)+str(ifig+1)))
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
            plt.title(self.names[exp])

        plt.tight_layout()

    def plot_relative_vorticity_snapshot(self, exps, Time=-1, zl=0):
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
            ax[ifig].set_title(self.names[exp])

        ax[0].set_ylabel('Latitude')
        if (yfig>1):
            ax[xfig].set_ylabel('Latitude')
        
        fig.colorbar(p, ax=ax, label='N/D units')

    def plot_KE_snapshot(self, exps, Time=-1, zl=0):
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

    def plot_EKE(self, exps, tstart=3650., zl=0, vmax = 0.02):
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
            ax[ifig].set_title(self.names[exp])

        ax[0].set_ylabel('Latitude')
        if (yfig>1):
            ax[xfig].set_ylabel('Latitude')
        
        fig.colorbar(p, ax=ax, label='$m^2/s^2$')

    def plot_KE_spectrum(self, exps, tstart = 7200., **kw):
        fig = plt.figure(figsize=(13,5))
        plt.rcParams.update({'font.size': 16})
        for exp in exps:
            prog = self[exp].prog
            t = prog.Time
            u = np.array(prog.u[t >= tstart])
            v = np.array(prog.v[t >= tstart])

            uh = 0.5 * (u[:,:,:,1:] + u[:,:,:,0:-1])
            vh = 0.5 * (v[:,:,1:,:] + v[:,:,0:-1,:])

            plt.subplot(121)
            k, Eu = compute_spectrum(uh[:,0,:,:], **kw)
            k, Ev = compute_spectrum(vh[:,0,:,:], **kw)
            plt.loglog(k,Eu+Ev, label=self.names[exp])
            plt.xlabel('$k$, wavenumber')
            plt.ylabel('$E(k)$')
            plt.title('Upper layer')
            plt.xlim((1,800))
            plt.legend()

            plt.subplot(122)
            k, Eu = compute_spectrum(uh[:,1,:,:], **kw)
            k, Ev = compute_spectrum(vh[:,1,:,:], **kw)
            plt.loglog(k,Eu+Ev, label=self.names[exp])
            plt.xlabel('$k$, wavenumber')
            plt.ylabel('$E(k)$')
            plt.title('Lower layer')
            plt.xlim((1,800))

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

        plt.tight_layout()

    def plot_cospectrum(self, exps, tstart = 7200., print_diagnostics = False, **kw):
        fig = plt.figure(figsize=(13,5))
        plt.rcParams.update({'font.size': 16})
        for exp in exps:
            prog = self[exp].prog
            mom = self[exp].mom

            t = prog.Time
            u = np.array(prog.u[t >= tstart])
            v = np.array(prog.v[t >= tstart])
            fx = np.array(mom.diffu[t >= tstart])
            fy = np.array(mom.diffv[t >= tstart])
            h = np.array(prog.h[t >= tstart])

            uh = 0.5 * (u[:,:,:,1:] + u[:,:,:,0:-1])
            vh = 0.5 * (v[:,:,1:,:] + v[:,:,0:-1,:])
            fxh = 0.5 * (fx[:,:,:,1:] + fx[:,:,:,0:-1])
            fyh = 0.5 * (fy[:,:,1:,:] + fy[:,:,0:-1,:])

            if print_diagnostics:
                print('Check integrals:')
                print('mean(u * fx + v * fy) = ', 
                    np.nanmean(uh*fxh, axis=(0,2,3)) +
                    np.nanmean(vh*fyh, axis=(0,2,3)))
                print('mean(h * u * fx + h * v * fy) / mean(h) = ', 
                    np.nanmean(h*uh*fxh, axis=(0,2,3)) / np.nanmean(h, axis=(0,2,3)) + 
                    np.nanmean(h*vh*fyh, axis=(0,2,3)) / np.nanmean(h, axis=(0,2,3)))

            plt.subplot(121)
            k, E = compute_cospectrum_uv(uh[:,0,:,:], vh[:,0,:,:], fxh[:,0,:,:], fyh[:,0,:,:], **kw)
            plt.semilogx(k,E*k, label=self.names[exp])
            plt.xlabel('$k$, wavenumber')
            plt.ylabel(r'$k \oint Re(\mathbf{u}_k \mathbf{f}_k^*) dk$')
            plt.title('Upper layer')
            if print_diagnostics:
                print('Integral upper layer:', E.sum() * (k[1]-k[0]))

            plt.subplot(122)
            k, E = compute_cospectrum_uv(uh[:,1,:,:], vh[:,1,:,:], fxh[:,1,:,:], fyh[:,1,:,:], **kw)
            plt.semilogx(k,E*k, label=self.names[exp])
            plt.xlabel('$k$, wavenumber')
            plt.title('Lower layer')
            plt.legend()
            if print_diagnostics:
                print('Integral lower layer:', E.sum() * (k[1]-k[0]))

        plt.tight_layout()

    def plot_cospectrum_componentwise(self, exps, tstart = 7200., print_diagnostics = False, **kw):
        fig = plt.figure(figsize=(13,5))
        plt.rcParams.update({'font.size': 16})
        for exp in exps:
            prog = self[exp].prog
            mom = self[exp].mom

            t = prog.Time
            u = np.array(prog.u[t >= tstart])
            v = np.array(prog.v[t >= tstart])
            fx = np.array(mom.diffu[t >= tstart])
            fy = np.array(mom.diffv[t >= tstart])
            h = np.array(prog.h[t >= tstart])
            ZBx = np.array(mom.ZB2020u[t >= tstart])
            ZBy = np.array(mom.ZB2020v[t >= tstart])

            uh = 0.5 * (u[:,:,:,1:] + u[:,:,:,0:-1])
            vh = 0.5 * (v[:,:,1:,:] + v[:,:,0:-1,:])
            fxh = 0.5 * (fx[:,:,:,1:] + fx[:,:,:,0:-1])
            fyh = 0.5 * (fy[:,:,1:,:] + fy[:,:,0:-1,:])
            ZBxh = 0.5 * (ZBx[:,:,:,1:] + ZBx[:,:,:,0:-1])
            ZByh = 0.5 * (ZBy[:,:,1:,:] + ZBy[:,:,0:-1,:])
            
            plt.subplot(121)
            k, E = compute_cospectrum_uv(uh[:,0,:,:], vh[:,0,:,:], fxh[:,0,:,:], fyh[:,0,:,:], **kw)
            k, EZB = compute_cospectrum_uv(uh[:,0,:,:], vh[:,0,:,:], ZBxh[:,0,:,:], ZByh[:,0,:,:], **kw)
            Esmag = E - EZB
            plt.semilogx(k,E*k, label='sum')
            plt.semilogx(k,EZB*k, '--', label='ZB2020')
            plt.semilogx(k,Esmag*k, '-.', label='Smag')
            plt.axhline(y=0,color='k', linestyle='--', alpha=0.5)
            plt.xlabel('$k$, wavenumber')
            plt.ylabel(r'$k \oint Re(\mathbf{u}_k \mathbf{f}_k^*) dk$')
            plt.title('Upper layer')
            if print_diagnostics:
                print('Integral upper layer:', E.sum() * (k[1]-k[0]))

            plt.subplot(122)
            k, E = compute_cospectrum_uv(uh[:,1,:,:], vh[:,1,:,:], fxh[:,1,:,:], fyh[:,1,:,:], **kw)
            k, EZB = compute_cospectrum_uv(uh[:,1,:,:], vh[:,1,:,:], ZBxh[:,1,:,:], ZByh[:,1,:,:], **kw)
            Esmag = E - EZB
            plt.semilogx(k,E*k, label='sum')
            plt.semilogx(k,EZB*k, '--', label='ZB2020')
            plt.semilogx(k,Esmag*k, '-.', label='Smag')
            plt.axhline(y=0,color='k', linestyle='--', alpha=0.5)
            plt.xlabel('$k$, wavenumber')
            plt.title('Lower layer')
            plt.legend()
            if print_diagnostics:
                print('Integral lower layer:', E.sum() * (k[1]-k[0]))

        plt.tight_layout()

    def plot_SGS_snapshot(self, exp, Time = -1):
        fig = plt.figure(figsize=(15,7.5))
        plt.rcParams.update({'font.size': 16})
        
        mom = self[exp].mom
        fx = mom.diffu.isel(Time=Time)
        fy = mom.diffv.isel(Time=Time)
        ZBx = mom.ZB2020u.isel(Time=Time)
        ZBy = mom.ZB2020v.isel(Time=Time)
        smagx = fx - ZBx
        smagy = fy - ZBy

        plt.subplot(241)
        plt.imshow(smagx.isel(zl=0), origin='lower', cmap='bwr')
        plt.title('diffu')
        plt.ylabel('Upper Layer')
        plt.colorbar()
        plt.clim(-1e-7, 1e-7)

        plt.subplot(242)
        plt.imshow(smagy.isel(zl=0), origin='lower', cmap='bwr')
        plt.title('diffv')
        plt.colorbar()
        plt.clim(-1e-7, 1e-7)

        plt.subplot(243)
        plt.imshow(ZBx.isel(zl=0), origin='lower', cmap='bwr')
        plt.title('ZB2020u')
        plt.colorbar()
        plt.clim(-1e-7, 1e-7)

        plt.subplot(244)
        plt.imshow(ZBy.isel(zl=0), origin='lower', cmap='bwr')
        plt.title('ZB2020v')
        plt.colorbar()
        plt.clim(-1e-7, 1e-7)

        plt.subplot(245)
        plt.imshow(smagx.isel(zl=1), origin='lower', cmap='bwr')
        plt.title('diffu')
        plt.ylabel('Lower Layer')
        plt.colorbar()
        plt.clim(-1e-7, 1e-7)

        plt.subplot(246)
        plt.imshow(smagy.isel(zl=1), origin='lower', cmap='bwr')
        plt.title('diffv')
        plt.colorbar()
        plt.clim(-1e-7, 1e-7)

        plt.subplot(247)
        plt.imshow(ZBx.isel(zl=1), origin='lower', cmap='bwr')
        plt.title('ZB2020u')
        plt.colorbar()
        plt.clim(-1e-7, 1e-7)

        plt.subplot(248)
        plt.imshow(ZBy.isel(zl=1), origin='lower', cmap='bwr')
        plt.title('ZB2020v')
        plt.colorbar()
        plt.clim(-1e-7, 1e-7)

        plt.tight_layout()

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