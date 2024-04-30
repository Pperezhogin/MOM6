import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import torch
import xrft
import gsw
from xgcm.padding import pad as xgcm_pad
from functools import lru_cache

from helpers.ann_tools import image_to_nxn_stencil_gpt, import_ANN, torch_pad, tensor_from_xarray
from helpers.selectors import select_LatLon, x_coord, y_coord
import warnings
warnings.filterwarnings("ignore")

def Coriolis(lat, compute_beta=False):
    '''
    Input (lat) here is in degrees, but later converted
    to radians

    f = 2 * Omega * sin (lat) is the Coriolis parameter 
    with Omega = 7.2921 x 10-5 rad/s being Earth rotation
    (https://en.wikipedia.org/wiki/Coriolis_frequency)

    beta = df/dy = 2 * Omega / R_e * cos (lat), R_e is the Earth radius 6.371e+6 m 
    (https://ceoas.oregonstate.edu/rossby_radius)
    '''
    Omega = 7.2921e-5
    lat_rad = lat * np.pi / 180. # latitude in radians
    f = 2 * Omega * np.sin(lat_rad)

    if compute_beta:
        R_e = 6.371e+6
        beta = 2 * Omega * np.cos(lat_rad) / R_e
        return f, beta
    else:
        return f

def vertical_modes_one_column_WKB(N2, dzB, dzT, N2_small=1e-8, 
                   dirichlet_surface=False, dirichlet_bottom=False,
                   few_modes=1,
                   SQG=False, scales=[1e+1,1e+2,1e+3], **kw):
    '''
    WKB approximations for the vertical modes of vertical_modes_one_column
    with lowest order accuracy.

    The central element of WKB approximation is to introduce the stretched 
    vertical coordinate (z_s = int(N(z'),z'=-z..0) / int(N(z'),z'=-H..0)) in (0,1)
    where 1 corresponds to bottom and 0 corresponds to surface
    '''

    if len(N2) != len(dzB):
        print('Error: len(N2) != len(dzB)')
    if len(N2) != len(dzT)-1:
        print('Error: len(N2) != len(dzT)-1')

    N2_bound = np.maximum(N2, N2_small)
    N = np.sqrt(N2_bound)

    # Find the coordinate of centerpoints
    # To be improved later
    zl = dzT * 0
    zl[0] = dzB[0] - dzT[0]/2
    zl[1:] = np.cumsum(dzB)

    # We first compute normalization in the denominator
    normalization = np.sum(N * dzB)

    # Then instead of sum we do cumulative sum
    # note that the number of elements does not change
    z_s = np.cumsum(N * dzB) / normalization
    
    # Now we need to increase the number of points. We place the boundary condition:
    # zero at first element
    z_s = np.pad(z_s, (1,0))

    if SQG:
        modes = []
        for scale in scales:
            mode = np.exp(-z_s * np.sqrt(scale) * normalization)
            modes.append(mode)
    else:
        if dirichlet_surface and dirichlet_bottom:
            frequencies = (1+np.arange(few_modes)) * np.pi
            modes = []
            for freq in frequencies:
                mode = np.sin(freq * z_s)
                modes.append(mode)
        
        if not(dirichlet_surface) and not(dirichlet_bottom):
            frequencies = (1+np.arange(few_modes)) * np.pi
            modes = []
            for freq in frequencies:
                mode = np.cos(freq * z_s)
                modes.append(mode)

        if not(dirichlet_surface) and dirichlet_bottom:
            frequencies = (1/2+np.arange(few_modes)) * np.pi
            modes = []
            for freq in frequencies:
                mode = np.cos(freq * z_s)
                modes.append(mode)

        if dirichlet_surface and not(dirichlet_bottom):
            frequencies = (1/2+np.arange(few_modes)) * np.pi
            modes = []
            for freq in frequencies:
                mode = np.sin(freq * z_s)
                modes.append(mode)

    return np.stack(modes,-1), z_s

def vertical_modes_one_column(N2, dzB, dzT, N2_small=1e-8, 
                   dirichlet_surface=False, dirichlet_bottom=False,
                   debug=False, few_modes=1,
                   SQG=False, scales=[1e+1,1e+2,1e+3], high_order_dirichlet=True):
    '''
    First try it. Figure 2b from Wenda Zhang 2024 
    "The role of surface potential vorticity in the vertical structure of mesoscale eddies in wind-driven ocean circulations":

        Npoints = 100
        zi = np.linspace(0,-1,Npoints)
        zl = (zi[0:-1] + zi[1:])/2
        dz = -np.diff(zi)[0]
        zi = zi[1:-1]
        
        N2 = np.exp(5*zi)
        dzB = dz * np.ones(Npoints-2)
        dzT = dz * np.ones(Npoints-1)

        plt.subplot(1,2,1)
        plt.plot(N2,zi,lw=3)
        plt.ylabel('Depth')
        plt.title('Stratification profile, $N^2$')
        plt.grid()
        plt.xlim([-1.1,1.1])
        plt.ylim([-1.1,0.1])

        plt.subplot(1,2,2)
        modes, cg = vertical_modes_one_column(N2, dzB, dzT, dirichlet_surface=True, dirichlet_bottom=False, few_modes=3)
        plt.plot(modes,zl,lw=3)
        plt.ylabel('Depth')
        plt.title('W. Zhang 2024 (Interior modes)')
        plt.grid()
        plt.xlim([-1.1,1.1])
        plt.ylim([-1.1,0.1])

        plt.tight_layout()

    Solves the eigenvalue problem:
    d/dz (1/N^2 d phi/dz) = -lambda^2 phi
    with Neuman (dphi/dz=0) or Dirichlet (phi=0) 
    boundary conditions, see Smith 2007
    "The geography of linear baroclinic instability in Earth oceans"
    https://elischolar.library.yale.edu/cgi/viewcontent.cgi?article=1178&context=journal_of_marine_research

    Inputs:
    N2 [1/s^2] - 1D numpy array of Nsquared defined on interfaces of finite volumes,
    where the upper interface is free surface and the lower interface is the bottom.
    Note: first and last points (surface and bottom) are not
    used for computations, and thus are NOT PROVIDED

    dzT [m] - grid steps in cell centers
    dzB [m] - grid steps in cell interfaces

    len(N2) = len(dzB),
    len(N2) = len(dzT) - 1

    dirichlet_surface - use Dirichlet B.C. at the surface, otherwise Neumann is used
    dirichlet_bottom  - use Dirichlet B.C. at the bottom,  otherwise Neumann is used

    few_modes - the number of modes to return

    N2_small = 1e-8 1/s^2 is a small threshould 
    for Nsquared is given by Chelton 1998, see Appendix C.a:

    Additional feature to solve for Surface-Trapped (SQG) mode,
    see Wenda Zhang 2024:
    B.C. will be applied automatically, and scale parameter
    "scale">0 will be used
    
    Output:
    First (or few) modes phi
    defined in centers of finite volumes, while
    N2 is defined in interfaces BETWEEN cells,
    so for one mode we have
    len(N2) = len(phi) - 1
    
    Together with mode, we return velocity of internal gravity wave
    cg = 1 / lambda or in terms of eigenvalues cg = sqrt(-1/eigenValue)

    Algorithm:
    * Find bottom by considering mask
    * Compute grid steps in centers (dzT) and interfaces (dzB)
    * Form 1D arrays to be used as diagonals of the discretization:
        * d1 = 1 / dzT (outer derivative)
        * d2 = 1 / N^2 / dzB (innder derivative and weighting with frequency)
        Note: len(d1) = len(d2) + 1
    * Form outder derivative weighting matrix:
        D1 = np.diag(d1)
    * Form matrix of second derivative (up to D1). Below zero Neumann B.C.:
        D2 = [
               -d2[0],  d2[0],       0
                d2[0], -d2[0]-d2[1], d2[1]
                0,      d2[1],      -d2[1]     
                ]
    * Total matrix of second derivatives (zero Neuman B.C.):
        D = D1 @ D2
    * In case of zero Dirichlet B.C. replace matrix D2 with:
        D2 = [
             -2*d2[0],  d2[0],       0
                d2[0], -d2[0]-d2[1], d2[1]
                0,      d2[1],    -2*d2[1]     
                ]

    Test assembling of matrix and its properties:
    N2 = np.arange(3)+1
    dzB = np.ones(3)
    dzT = np.ones(4)
    D, eigenValues, eigenVectors = vertical_modes_one_column(N2, dzB, dzT, dirichlet_surface=True, dirichlet_bottom=True, debug=True)
    '''
    if len(N2) != len(dzB):
        print('Error: len(N2) != len(dzB)')
    if len(N2) != len(dzT)-1:
        print('Error: len(N2) != len(dzT)-1')

    if SQG:
        dirichlet_surface = True
        dirichlet_bottom = False
    
    N2_bound = np.maximum(N2, N2_small)

    d1 = 1 / dzT
    d2 = 1 / (N2_bound * dzB)

    D1 = np.diag(d1)
    D2 = - np.diag(np.pad(d2,(0,1))) - np.diag(np.pad(d2,(1,0)))
    D2 += np.diag(d2,1) + np.diag(d2,-1)

    # Higher order method accounts for the grid staggering.
    # Here we introduce the ghost cell with B.C.:
    # phi_{-1} + phi_{0} = 0
    # This gives 3 instead of 2
    if high_order_dirichlet:
        TwoOrThree = 3
    else:
        TwoOrThree = 2

    if dirichlet_surface:
        D2[0,0] *= TwoOrThree
    if dirichlet_bottom:
        D2[-1,-1] *= TwoOrThree

    D = D1@D2

    if not(SQG):
        # Compute eigenvalues and eigenvectors, and sort in descending order
        eigenValues, eigenVectors = np.linalg.eig(D)
        idx = eigenValues.argsort()[::-1]   
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]

        if debug:
            print('Eigenvalues in descending order:', eigenValues)
            print('Error in Eigendecomposition norm(D - V*L*V^T)/norm(D):',
                np.linalg.norm(eigenVectors @ np.diag(eigenValues) @ eigenVectors.T - D) / np.linalg.norm(D))
            print('Error in eigenvectors orthogonality: norm(V*V^T-E)/norm(E), norm(V^T*V-E)/norm(E):',
                np.linalg.norm(eigenVectors @ eigenVectors.T-np.eye(len(d1))), np.linalg.norm(eigenVectors.T @ eigenVectors-np.eye(len(d1))))
            print('Every vector is properly shaped: norm(D*v_1 - eigval_1*v_1)/norm(v_1):',
                np.linalg.norm(D @ eigenVectors[:,0] - eigenValues[0] * eigenVectors[:,0]) / np.linalg.norm(eigenVectors[:,0]))
            return D, eigenValues, eigenVectors

        # filter out barotropic mode
        if not dirichlet_bottom and not dirichlet_surface:
            shift = 1
        else:
            shift = 0

        # Select few first modes
        modes = eigenVectors[:,shift:few_modes+shift]
        # Internal gravity wave velocities
        # cg = sqrt(-1/eigenValue)
        cg = np.sqrt(-1/eigenValues[shift:few_modes+shift])

        # Normalize modes such that surface value is positive
        # and maximum value is 1
        modes = modes / np.max(np.abs(modes),0) * np.sign(modes[0,:])

        return modes, cg
    else:
        modes = []
        for scale in scales:
            # Here we construct new matric for the operator
            # d/dz (1/N^2 d phi/dz) - scale * phi
            E = np.eye(len(d1))
            # We should not disturb the Neuman boundary condition
            E[-1,-1] = 0.
            M = D - scale * E
            
            # Here we place the dirichlet B.C. with value phi=1:
            rhs = np.zeros(len(d1))
            if high_order_dirichlet:
                # This correction comes from the equation
                # phi[-1] + phi[0] = 2 * phi[-1/2]
                rhs[0] = - 2 * d2[0] * d1[0]    
            else:
                rhs[0] = - d2[0] * d1[0]
            mode = np.linalg.solve(M, rhs)
            modes.append(mode)
        return np.stack(modes,-1)

class StateFunctions():
    def __init__(self, data, param, grid):
        self.data = data
        self.param = param
        self.grid = grid
    
    def __del__(self):
        del self.data
        del self.param
        del self.grid
        
    def sample_grid_harmonic(self, grid_harmonic='chess_vorticity'):
        '''
        Available grid harmonics to sample:
        'chess_vorticity'
        'chess_divergence'
        'plane_wave'
        
        Return new object StateFunctions with data containing waves
        '''
        data = self.data[['u','v']].copy()
        ny, nx = data.u.shape

        if grid_harmonic in ['chess_vorticity', 'chess_divergence']:
            u = np.zeros((ny,nx))
            v = np.zeros((ny,nx))
            
            # assign random phase (1 or -1)
            phase = -1 if np.random.randint(2)==0 else 1

            if grid_harmonic == 'chess_vorticity':
                sign = -1
            elif grid_harmonic == 'chess_divergence':
                sign = 1
            else:
                print('Error: grid_harmonic is wrongly specified')

            for j in range(ny):
                for i in range(nx):
                    idx = i+j
                    u[j,i] = phase *        (-1)**(idx)
                    v[j,i] = phase * sign * (-1)**(idx)

        elif grid_harmonic == 'plane_wave':
            freq_x = 0
            freq_y = 0
            while np.abs(freq_x)<2/3*np.pi and np.abs(freq_y)<2/3*np.pi:
                freq_x = np.random.rand() * np.pi * 2 - np.pi
                freq_y = np.random.rand() * np.pi * 2 - np.pi

            phase_u = np.random.rand()*2*np.pi
            phase_v = np.random.rand()*2*np.pi

            i = np.ones(ny).reshape(-1,1)@np.arange(nx).reshape(1,-1)
            j = np.arange(ny).reshape(-1,1)@np.ones(nx).reshape(1,-1)

            u = np.sin(freq_x * i + freq_y * j + phase_u)
            v = np.sin(freq_x * i + freq_y * j + phase_v)
        else:
            print('Error: wrong grid harmonic')
                
        data['u'] = xr.DataArray(u, dims=['yh', 'xq']) * self.param.wet_u
        data['v'] = xr.DataArray(v, dims=['yq', 'xh']) * self.param.wet_v
        
        return StateFunctions(data, self.param, self.grid)
    
    def EZ_source_ANN(self, ann_Txy=None, ann_Txx_Tyy=None):
        ann = self.ANN(ann_Txy, ann_Txx_Tyy)
        
        return self.compute_EZ_source(
            ann['ZB20u'], ann['ZB20v'], ann['Txy'], ann['Txx'], ann['Tyy'])
        
    def compute_EZ_source(self, fx, fy, Txy=None, Txx=None, Tyy=None):
        '''
        Compute local and global sources of energy
        and enstrophy for a given forcing
        '''
        data = self.data
        param = self.param
        grid = self.grid

        areaT = param.dxT * param.dyT
        areaU = param.dxCu * param.dyCu
        areaV = param.dxCv * param.dyCv
        areaB = param.dxBu * param.dyBu
    
        # Energy source total
        Ex = fx * data.u
        Ey = fy * data.v
        dEdt = (grid.interp(Ex * areaU,'X') + grid.interp(Ey * areaV,'Y')) * param.wet / areaT
        
        # Energy source Galilean-invariant form
        if Txy is not None:
            sh_xy, sh_xx, vort_xy, _ = self.velocity_gradients()
            dEdt_G = - 0.5 * ((Txx * sh_xx - Tyy * sh_xx) * areaT + \
                grid.interp(2 * Txy * sh_xy * areaB,['X','Y']))
            dEdt_G = dEdt_G * param.wet / areaT
        else:
            dEdt_G = None

        # Enstrophy source
        f = self.relative_vorticity(fx,fy)
        vorticity = self.relative_vorticity(data.u,data.v)
        dZdt = grid.interp(f * vorticity * areaB, ['X', 'Y']) * param.wet / areaT
        
        # Palinstrophy source
        def gradient(w):
            wx = grid.diff(w, 'X') / param.dxCv * param.wet_v
            wy = grid.diff(w, 'Y') / param.dyCu * param.wet_u
            return wx, wy
        
        px, py = gradient(f.astype('float64'))
        wx, wy = gradient(vorticity)
        Px = px * wx * areaV
        Py = py * wy * areaU
        dPdt = (grid.interp(Px,'Y') + grid.interp(Py,'X')) * param.wet / areaT

        return {'dEdt_G': dEdt_G, 'dEdt_local': dEdt, 'dZdt_local': dZdt, 'dPdt_local': dPdt,
                'dEdt': float((dEdt*areaT).sum()), 'dZdt': float((dZdt*areaT).sum()), 'dPdt': float((dPdt*areaT).sum())}    
           
    def transfer(self, fu_in, fv_in,
            region = 'NA', window='hann', 
            nfactor=2, truncate=False, detrend='linear', 
            window_correction=True, compensated=True, 
            additional_spectra=False):
        '''
        This function computes energy transfer spectrum
        and optionally outputs the spatial and temporal power spectrum
        and KE spectrum
        '''
        
        if region == 'NA':
            kw = {'Lat': (25,45), 'Lon': (-60,-40)}
        elif region == 'Pacific':
            kw = {'Lat': (25,45), 'Lon': (-200,-180)}
        elif region == 'Equator':
            kw = {'Lat': (-30,30), 'Lon': (-190,-130)}
        elif region == 'ACC':
            kw = {'Lat': (-70,-30), 'Lon': (-40,0)}
        else:
            print('Error: wrong region')
            
        # Select desired Lon-Lat square
        u = select_LatLon(self.data.u,time=slice(None,None),**kw)
        v = select_LatLon(self.data.v,time=slice(None,None),**kw)
        fu = select_LatLon(fu_in,time=slice(None,None),**kw)
        fv = select_LatLon(fv_in,time=slice(None,None),**kw)
        
        if u.shape != v.shape:
            nx = min(len(x_coord(u)), len(x_coord(v)))
            ny = min(len(y_coord(u)), len(y_coord(v)))
            def sel(x):
                return x[{x_coord(x).name: slice(0,nx), y_coord(x).name: slice(0,ny)}]
            u = sel(u)
            v = sel(v)
            fu = sel(fu)
            fv = sel(fv)

        # Average grid spacing (result in metres)
        dx = select_LatLon(self.param.dxT,**kw).mean().values
        dy = select_LatLon(self.param.dyT,**kw).mean().values

        # define uniform grid
        for variable in [u, fu]:
            variable['xq'] = dx * np.arange(len(u.xq))
            variable['yh'] = dy * np.arange(len(u.yh))
            
        for variable in [v, fv]:
            variable['xh'] = dx * np.arange(len(v.xh))
            variable['yq'] = dy * np.arange(len(v.yq))

        # In a case of dimensions are transposed differently
        fu = fu.transpose(*u.dims)
        fv = fv.transpose(*v.dims)

        Eu = xrft.isotropic_cross_spectrum(u, fu, dim=('xq','yh'), window=window, nfactor=nfactor, 
            truncate=truncate, detrend=detrend, window_correction=window_correction)
        Ev = xrft.isotropic_cross_spectrum(v, fv, dim=('xh','yq'), window=window, nfactor=nfactor, 
            truncate=truncate, detrend=detrend, window_correction=window_correction)
        
        E = np.real(Eu+Ev)
        E['freq_r'] = E['freq_r']*2*np.pi # because library returns frequencies, but not wavenumbers
        if compensated:
            E = E * E['freq_r']

        if additional_spectra:
            # Spatial power spectrum of subgrid forcing
            Pu = xrft.isotropic_power_spectrum(fu, dim=('xq','yh'), window=window, nfactor=nfactor, 
                truncate=truncate, detrend=detrend, window_correction=window_correction)
            Pv = xrft.isotropic_power_spectrum(fv, dim=('xh','yq'), window=window, nfactor=nfactor,
                truncate=truncate, detrend=detrend, window_correction=window_correction)

            P = (Pu+Pv)
            P['freq_r'] = P['freq_r']*2*np.pi

            # Spatial KE spectrum
            KEu = xrft.isotropic_power_spectrum(u, dim=('xq','yh'), window=window, nfactor=nfactor,
                truncate=True, detrend=detrend, window_correction=window_correction)                                   
            KEv = xrft.isotropic_power_spectrum(v, dim=('xh','yq'), window=window, nfactor=nfactor,
                truncate=True, detrend=detrend, window_correction=window_correction)
            KE = (KEu+KEv) * 0.5 # As KE spectrum is half the power density
            KE['freq_r'] = KE['freq_r']*2*np.pi

            # Time power spectrum of subgrid forcing
            try:
                # Here we try to convert cftime to day format, so that frequency will be in day^-1
                dt = np.diff(fu.time.dt.day).max()
                fu['time'] = np.arange(len(fu.time))*dt
                fv['time'] = np.arange(len(fv.time))*dt
                u['time'] = np.arange(len(u.time))*dt
                v['time'] = np.arange(len(v.time))*dt
            except:
                pass
            Ps_u = xrft.power_spectrum(fu.chunk({'time':-1}), dim=('time'), window=window, nfactor=nfactor,
                truncate=truncate, detrend=detrend, window_correction=window_correction).mean(dim=('xq','yh'))
            Ps_v = xrft.power_spectrum(fv.chunk({'time':-1}), dim=('time'), window=window, nfactor=nfactor,
                truncate=truncate, detrend=detrend, window_correction=window_correction).mean(dim=('xh','yq'))
            Ps = (Ps_u+Ps_v)
            # Convert 2-sided power spectrum to one-sided
            Ps = Ps[Ps.freq_time>0]

            # Time power spectrum of KE
            KEs_u = xrft.power_spectrum(u.chunk({'time':-1}), dim=('time'), window=window, nfactor=nfactor,
                truncate=truncate, detrend=detrend, window_correction=window_correction).mean(dim=('xq','yh'))
            KEs_v = xrft.power_spectrum(v.chunk({'time':-1}), dim=('time'), window=window, nfactor=nfactor,
                truncate=truncate, detrend=detrend, window_correction=window_correction).mean(dim=('xh','yq'))
            KEs = (KEs_u+KEs_v) * 0.5
            # Convert 2-sided power spectrum to one-sided
            KEs = KEs[KEs.freq_time>0]

        if additional_spectra:
            return E, P, KE, Ps, KEs
        else:
            return E
    
    def transfer_ANN(self, ann_Txy, ann_Txx_Tyy, kw_ann={}, kw_sp={}):
        ann = self.ANN(ann_Txy, ann_Txx_Tyy, **kw_ann)
        return self.transfer(ann['ZB20u'], ann['ZB20v'], **kw_sp)
    
    def velocity_gradients(self, compute=False):
        param = self.param
        data = self.data
        grid = self.grid

        if compute:
            compute = lambda x: x.compute()
        else:
            compute = lambda x: x
        
        dudx = grid.diff(data.u * param.wet_u / param.dyCu, 'X') * param.dyT / param.dxT
        dvdy = grid.diff(data.v * param.wet_v / param.dxCv, 'Y') * param.dxT / param.dyT

        dudy = compute(grid.diff(data.u * param.wet_u / param.dxCu, 'Y') * param.dxBu / param.dyBu * param.wet_c)
        dvdx = compute(grid.diff(data.v * param.wet_v / param.dyCv, 'X') * param.dyBu / param.dxBu * param.wet_c)
        
        sh_xx = compute((dudx-dvdy) * param.wet)
        sh_xy = dvdx+dudy
        vort_xy=dvdx-dudy
        div = (dudx+dvdy) * param.wet # For VGM model
        
        return sh_xy, sh_xx, vort_xy, div
    
    def Smagorinsky(self, Cs_biharm=0.06):
        sh_xy, sh_xx, vort_xy, _ = self.velocity_gradients()
        grid = self.grid
        param = self.param
        
        # In center point
        Shear_mag = param.wet * (sh_xx**2+grid.interp(sh_xy**2,['X','Y']))**0.5
        
        # Biharmonic viscosity coefficient
        dx2h = param.dxT**2
        dy2h = param.dyT**2
        grid_sp2 = (2 * dx2h * dy2h) / (dx2h + dy2h)
        Biharm_const = Cs_biharm * grid_sp2**2
        
        # Convert to laplacian viscosity
        # using Griffies formula
        # nu_biharm = nu_lap * dx**2 / 8.
        Lap_const = Biharm_const * 8. / grid_sp2
        
        # Compute viscosity (i.e., harmonic one)
        viscosity = Lap_const * Shear_mag
        
        # There is no minus here because
        # we consider sign as 
        # du/dt = div(T)
        Txx = sh_xx * viscosity * param.wet
        Tyy = - Txx # There is no trace part
        Txy = sh_xy * grid.interp(viscosity,['X','Y']) * param.wet_c
        
        smagx = param.wet_u * (grid.diff(Txx*param.dyT**2, 'X') / param.dyCu     \
               + grid.diff(Txy*param.dxBu**2, 'Y') / param.dxCu) \
               / (param.dxCu*param.dyCu)

        smagy = param.wet_v * (grid.diff(Txy*param.dyBu**2, 'X') / param.dyCv     \
                   + grid.diff(Tyy*param.dxT**2, 'Y') / param.dxCv) \
                   / (param.dxCv*param.dyCv)
        
        return {'Txx': Txx, 'Tyy': Tyy, 'Txy': Txy, 'Shear_mag': Shear_mag, 'sh_xx': sh_xx, 'sh_xy': sh_xy, 'vort_xy': vort_xy, 'smagx': smagx, 'smagy': smagy}
        
    def ZB20(self, ZB_scaling=1.0, VGM='False'):
        param = self.param
        grid = self.grid
        
        sh_xy, sh_xx, vort_xy, div = self.velocity_gradients()

        vort_xy_center = grid.interp(vort_xy,['X','Y']) * param.wet
        sh_xy_center = grid.interp(sh_xy,['X','Y']) * param.wet
        sh_xx_corner = grid.interp(sh_xx,['X', 'Y']) * param.wet_c

        vort_sh = vort_xy_center * sh_xy_center

        sum_sq = 0.5 * (vort_xy_center**2 + sh_xy_center**2 + sh_xx**2)

        Txx = - vort_sh + sum_sq
        Tyy = + vort_sh + sum_sq
        Txy = vort_xy * sh_xx_corner

        if VGM == 'VGM_flux':
            div_corner = grid.interp(div, ['X', 'Y']) * param.wet_c
            Txx +=   div * sh_xx + div**2
            Tyy += - div * sh_xx + div**2
            Txy +=   div_corner * sh_xy

        kappa_t = - param.dxT * param.dyT * param.wet * ZB_scaling
        kappa_q = - param.dxBu * param.dyBu * param.wet_c * ZB_scaling

        Txx = kappa_t * Txx
        Tyy = kappa_t * Tyy
        Txy = kappa_q * Txy

        ZB20u = param.wet_u * (grid.diff(Txx*param.dyT**2, 'X') / param.dyCu     \
               + grid.diff(Txy*param.dxBu**2, 'Y') / param.dxCu) \
               / (param.dxCu*param.dyCu)

        ZB20v = param.wet_v * (grid.diff(Txy*param.dyBu**2, 'X') / param.dyCv     \
                   + grid.diff(Tyy*param.dxT**2, 'Y') / param.dxCv) \
                   / (param.dxCv*param.dyCv)
        
        if VGM == 'VGM_full':
            kappa_u = - param.dxCu * param.dyCu * param.wet_u * ZB_scaling
            kappa_v = - param.dxCv * param.dyCv * param.wet_v * ZB_scaling
            
            Dxy = grid.diff(sh_xx, 'X') / param.dxCu + grid.diff(sh_xy, 'Y') / param.dyCu
            div_u = grid.interp(div, 'X') * param.wet_u
            vort_xy_u = grid.interp(vort_xy, 'Y')
            div_y = grid.interp(grid.diff(div, 'Y') / param.dyCv,['X', 'Y'])
            ZB20u += kappa_u * (div_u * Dxy + vort_xy_u * div_y)

            Dxy = -grid.diff(sh_xx, 'Y') / param.dyCv + grid.diff(sh_xy, 'X') / param.dxCv
            div_v = grid.interp(div, 'Y') * param.wet_v
            vort_xy_v = grid.interp(vort_xy, 'X')
            div_x = grid.interp(grid.diff(div, 'X') / param.dxCu,['X', 'Y'])
            ZB20v += kappa_v * (div_v * Dxy - vort_xy_v * div_x)

        return {'ZB20u': ZB20u, 'ZB20v': ZB20v, 
                'Txx': Txx, 'Tyy': Tyy, 'Txy': Txy}
    
    @lru_cache(maxsize=1)
    def compute_features(self):
        '''
        Do all computations with xarrays so rest of the inference involves torch only.
        If inference happens multiple times, caching helps to prevent recomputing
        of features
        '''
        grid = self.grid
        param = self.param

        ########### Convert grid to torch #############
        wet = tensor_from_xarray(param.wet)
        wet_u = tensor_from_xarray(param.wet_u)
        wet_v = tensor_from_xarray(param.wet_v)
        wet_c = tensor_from_xarray(param.wet_c)
        dyT = tensor_from_xarray(param.dyT)
        dxT = tensor_from_xarray(param.dxT)
        dxCu = tensor_from_xarray(param.dxCu)
        dyCu = tensor_from_xarray(param.dyCu)
        dyCv = tensor_from_xarray(param.dyCv)
        dxCv = tensor_from_xarray(param.dxCv)
        dxBu = tensor_from_xarray(param.dxBu)
        dyBu = tensor_from_xarray(param.dyBu)
        areaBu = dxBu * dyBu
        areaT = dxT * dyT
        areaCu = dxCu * dyCu
        areaCv = dxCv * dyCv
        
        ############# Computation of velocity gradients #############
        sh_xy, sh_xx, vort_xy, _ = self.velocity_gradients(compute=True)

        sh_xy_h = tensor_from_xarray(grid.interp(sh_xy, ['X','Y'])) * wet
        vort_xy_h = tensor_from_xarray(grid.interp(vort_xy, ['X','Y'])) * wet
        sh_xx_q = tensor_from_xarray(grid.interp(sh_xx, ['X','Y'])) * wet_c

        sh_xy = tensor_from_xarray(sh_xy)
        sh_xx = tensor_from_xarray(sh_xx)
        vort_xy = tensor_from_xarray(vort_xy)

        return sh_xy, sh_xx, vort_xy, sh_xy_h, vort_xy_h, sh_xx_q, \
               wet, wet_u, wet_v, wet_c,                           \
               dyT, dxT, dxCu, dyCu, dyCv, dxCv, dxBu, dyBu,       \
               areaBu, areaT, areaCu, areaCv

    def Apply_ANN(self, ann_Txy=None, ann_Txx_Tyy=None, stencil_size=3,
                  time_revers=False, rotation=0, reflect_x=False, reflect_y=False,
                  dimensional_scaling=True, strain_norm = 1e-6, flux_norm = 1e-3):
        '''
        The only input is the dataset itself.
        The output is predicted momentum flux in physical
        units in torch format, and its divergence
        '''
        if 'time' in self.data.dims:
            raise NotImplementedError("This operation is not implemented for many time slices. Use a single time.")
        if ann_Txy is None:
            ann_Txy = import_ANN('../trained_models/ANN_Txy_ZB.nc')
            print('Warning: Prediction from default ANN')
        if ann_Txx_Tyy is None:
            ann_Txx_Tyy = import_ANN('../trained_models/ANN_Txx_Tyy_ZB.nc')
            print('Warning: Prediction from default ANN')

        ########## Symmetries treatment ###########
        # Rotation symmetry
        if rotation in [0, 180]:
            rotation_sign = 1
        elif rotation in [90, 270]:
            rotation_sign = -1
        else:
            print('Error: use rotation one of 0, 90, 180, 270')
        
        # Reflection symmetry
        reflect_sign = 1
        if reflect_x:
            reflect_sign = - reflect_sign
        if reflect_y:
            reflect_sign = - reflect_sign
        
        # Time reversibility symmetry
        if time_revers:
            reverse_sign = -1
        else:
            reverse_sign = 1

        ############# Helper functions ################
        def norm(x):
            '''
            Norm is computed with double precision to prevent overflow
            '''
            return torch.sqrt((x.type(torch.float64)**2).sum(dim=-1, keepdims=True)).type(torch.float32)

        def extract_nxn(x):
            y = torch_pad(x, one_side_pad=stencil_size//2, left=True, right=True, top=True, bottom=True)
            return image_to_nxn_stencil_gpt(y, stencil_size=stencil_size,
                rotation=rotation, reflect_x=reflect_x, reflect_y=reflect_y)
        
        ############# Compute features in torch ###############
        sh_xy, sh_xx, vort_xy, sh_xy_h, vort_xy_h, sh_xx_q, \
        wet, wet_u, wet_v, wet_c,                           \
        dyT, dxT, dxCu, dyCu, dyCv, dxCv, dxBu, dyBu,       \
        areaBu, areaT, areaCu, areaCv = self.compute_features()
        
        ############# Prediction of Txy ###############
        # Collect input features
        input_features = torch.concat(
                        [
                        extract_nxn(sh_xy   * (rotation_sign * reflect_sign * reverse_sign)), 
                        extract_nxn(sh_xx_q * (rotation_sign * reverse_sign)), 
                        extract_nxn(vort_xy * (reflect_sign  * reverse_sign))
                        ],-1)

        # Normalize input features
        if dimensional_scaling:
            input_norm = norm(input_features)
            input_features = (input_features / (input_norm+1e-30))
        else:
            input_features = input_features / strain_norm

        # Make prediction with transforming prediction back to original frame
        Txy = ann_Txy(input_features) * (rotation_sign * reflect_sign * reverse_sign)

        # Now denormalize the output
        if dimensional_scaling:
            Txy = Txy * input_norm * input_norm * areaBu.reshape(-1,1)
        else:
            Txy = Txy * flux_norm
        
        # Apply BC. Minus sign is needed for consistency with ZB
        Txy = - Txy.reshape(wet_c.shape) * wet_c

        ########## Second, prediction of Txx, Tyy ###############
        input_features = torch.concat(
                        [
                        extract_nxn(sh_xy_h   * (rotation_sign * reflect_sign * reverse_sign)), 
                        extract_nxn(sh_xx     * (rotation_sign * reverse_sign)), 
                        extract_nxn(vort_xy_h * (reflect_sign  * reverse_sign))
                        ],-1)

        # Normalize input features
        if dimensional_scaling:
            input_norm = norm(input_features)
            input_features = (input_features / (input_norm+1e-30))
        else:
            input_features = input_features / strain_norm

        # Make prediction
        Tdiag = ann_Txx_Tyy(input_features) * reverse_sign

        # Now denormalize the output
        if dimensional_scaling:
            Tdiag = Tdiag * input_norm * input_norm * (areaT).reshape(-1,1)
        else:
            Tdiag = Tdiag * flux_norm
        
        # This transforms the prediction 
        # back to original frame
        if rotation in [0, 180]:
            Txx_idx = 0
            Tyy_idx = 1
        elif rotation in [90, 270]:
            Txx_idx = 1
            Tyy_idx = 0
        else:
            print('Error: use rotation one of 0, 90, 180, 270')
        
        Txx =  - Tdiag[:,Txx_idx].reshape(wet.shape) * wet
        Tyy =  - Tdiag[:,Tyy_idx].reshape(wet.shape) * wet
        
        Txx_padded = torch_pad(Txx * dyT**2, right=True)
        Txy_padded = torch_pad(Txy * dxBu**2, bottom=True)
        ZB20u = wet_u * (torch.diff(Txx_padded,dim=-1) / dyCu + torch.diff(Txy_padded,dim=-2) / dxCu) / (areaCu)
        
        Txy_padded = torch_pad(Txy * dyBu**2,left=True)
        Tyy_padded = torch_pad(Tyy * dxT**2, top=True)
        ZB20v = wet_v * (torch.diff(Txy_padded,dim=-1) / dyCv + torch.diff(Tyy_padded,dim=-2) / dxCv) / (areaCv)
        
        return {'Txx': Txx, 'Tyy': Tyy, 'Txy': Txy, 
                'ZB20u': ZB20u, 'ZB20v': ZB20v, 
                'sh_xx': sh_xx, 'sh_xy': sh_xy, 'vort_xy': vort_xy}
    
    def ANN(self, ann_Txy=None, ann_Txx_Tyy=None, stencil_size = 3,
            time_revers=False, rotation=0, reflect_x=False, reflect_y=False,
            dimensional_scaling=True, strain_norm = 1e-6, flux_norm = 1e-3):
        pred = self.Apply_ANN(ann_Txy, ann_Txx_Tyy, stencil_size,
                              time_revers, rotation, reflect_x, reflect_y,
                              dimensional_scaling, strain_norm, flux_norm)
        
        Txy = pred['Txy'].detach().numpy() + self.param.dxBu * 0
        Txx = pred['Txx'].detach().numpy() + self.param.dxT * 0
        Tyy = pred['Tyy'].detach().numpy() + self.param.dxT * 0
        ZB20u = pred['ZB20u'].detach().numpy() + self.param.dxCu * 0
        ZB20v = pred['ZB20v'].detach().numpy() + self.param.dxCv * 0

        return {'Txx': Txx, 'Tyy': Tyy, 'Txy': Txy, 
                'ZB20u': ZB20u, 'ZB20v': ZB20v,
                'sh_xx': pred['sh_xx'], 'sh_xy': pred['sh_xy'], 'vort_xy': pred['vort_xy']}
    
    def KE_Arakawa(self):
        '''
        https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L1000-L1003
        '''
        param = self.param
        u = self.data.u
        v = self.data.v
        grid = self.grid
        
        areaCu = param.dxCu * param.dyCu
        areaCv = param.dxCv * param.dyCv
        areaT = param.dxT * param.dyT
        
        KEu = grid.interp(param.wet_u * areaCu * u**2, 'X')
        KEv = grid.interp(param.wet_v * areaCv * v**2, 'Y')

        # Zero Neumann B.C. in a sense that there is no
        # overshoot near the boundary as a result of interpolation
        # Note: this is an adhoc approach which serves to resolve
        # inconsistency in data when filtered velocity is not zero
        # near the boundary as it happens in gcm_filters because 
        # they have only Neumann b.c. for filtered field
        wet_u = grid.interp(param.wet_u,'X')
        wet_v = grid.interp(param.wet_v,'Y')
        KEu = xr.where(wet_u>0, KEu / wet_u, 0.)
        KEv = xr.where(wet_v>0, KEv / wet_v, 0.)
        
        return 0.5 * (KEu + KEv) / areaT * param.wet

    def gradKE(self):
        '''
        https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L1029-L1034
        '''
        
        param = self.param
        grid = self.grid

        KE = self.KE_Arakawa()
        IdxCu = 1. / param.dxCu
        IdyCv = 1. / param.dyCv

        KEx = grid.diff(KE, 'X') * IdxCu * param.wet_u
        KEy = grid.diff(KE, 'Y') * IdyCv * param.wet_v
        return (KEx, KEy)

    def relative_vorticity(self, u=None, v=None):
        '''
        https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L472
        '''
        param = self.param
        grid = self.grid
        
        if u is None and v is None:
            u = self.data.u
            v = self.data.v
        
        dyCv = param.dyCv
        dxCu = param.dxCu
        IareaBu = 1. / (param.dxBu * param.dyBu)
        # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L309-L310
        dvdx = grid.diff(param.wet_v * v * dyCv,'X')
        dudy = grid.diff(param.wet_u * u * dxCu,'Y')
        return (dvdx - dudy) * IareaBu * param.wet_c
    
    def relative_vorticity_torch(self, u, v):
        '''
        Analog of the function above but for torch tensors
        Here we assume that u and v are torch tensors
        '''
        from torch.nn.functional import pad as torch_native_pad
        def tensor(x, torch_type=torch.float32):
            return torch.tensor(x.values).type(torch_type)
        
        param = self.param
        dyCv = tensor(param.dyCv)
        dxCu = tensor(param.dxCu)
        wet_u = tensor(param.wet_u)
        wet_v = tensor(param.wet_v)
        wet_c = tensor(param.wet_c)
        IareaBu = tensor(1. / (param.dxBu * param.dyBu))
        
        V = wet_v * v * dyCv
        U = wet_u * u * dxCu
        
        def zonal_circular_pad(x, right=True):
            y = torch.zeros(x.shape[-2], x.shape[-1]+1)
            if right:
                y[:,:-1] = x
                y[:,-1] = x[:,0]
            else:
                y[:,1:] = x
                y[:,0] = x[:,-1]
            return y
        
        V_padded = zonal_circular_pad(V, right=True)
        dvdx = V_padded[:,1:] - V_padded[:,:-1]
        U_padded = torch_native_pad(U, (0,0,0,1)) # pad on the right with zero along meridional 
        dudy = U_padded[1:,:] - U_padded[:-1,:]
        return (dvdx - dudy) * IareaBu * wet_c

    def PV_cross_uv(self):
        '''
        https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L669-L671
        https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L788-L790
        fx = + q * vh
        fy = - q * uh
        '''
        param = self.param
        u = self.data.u
        v = self.data.v
        grid = self.grid
        
        # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L131
        # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_continuity_PPM.F90#L569-L570
        uh = u * param.dyCu * param.wet_u
        # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L133
        vh = v * param.dxCv * param.wet_v
        # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L484
        rel_vort = self.relative_vorticity()

        # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L247
        Area_h = param.dxT * param.dyT
        # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L272-L273
        Area_q = grid.interp(Area_h, ['X', 'Y']) * 4
        # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L323
        hArea_u = grid.interp(Area_h,'X')
        # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L320
        hArea_v = grid.interp(Area_h,'Y')
        # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L488
        hArea_q = 2 * grid.interp(hArea_u,'Y') + 2 * grid.interp(hArea_v,'X')
        # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L489
        Ih_q = Area_q / hArea_q

        # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L490
        q = rel_vort * Ih_q

        IdxCu = 1. / param.dxCu
        IdyCv = 1. / param.dyCv
        # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L669-L671
        CAu = + grid.interp(q * grid.interp(vh,'X'),'Y') * IdxCu * param.wet_u
        # https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L788-L790
        CAv = - grid.interp(q * grid.interp(uh,'Y'),'X') * IdyCv * param.wet_v

        return (CAu, CAv)

    def advection(self):
        '''
        https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L751
        https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L875
        '''
        CAu, CAv = self.PV_cross_uv()
        KEx, KEy = self.gradKE()
        return (CAu - KEx, CAv - KEy)

    def rho(self, potential=True):
        '''
        Compute POTENTIAL density in kg/m^3, which is sigma0+1000kg/m^3.
        We compute sigma0 according to https://www.teos-10.org/pubs/gsw/html/gsw_sigma0.html
        Note that instead of Absolute Salinity and Conservative Temperature we use
        standard CM2.6 variables: Potential temperature, degrees C and Practical Salinity, psu
        '''
        if 'rho' in self.data.variables:
            if potential:
                return self.data.rho
            else:
                if not('temp' in self.data.variables):
                    print('Warning: returning potential density instead of in situ')
                    return self.data.rho

        if potential:
            rho_array = (1000.+gsw.sigma0(self.data.salt, self.data.temp))
        else:
            # Here we return in-situ density assuming that 
            # pressure in dbar equals z (this is a good approximation)
            # and same as above neglecting difference between potential and conservative
            # temperatures, and practical and absolute salinity
            rho_array = gsw.rho(self.data.salt, self.data.temp, self.param.zl)
        
        # Place b.c. with reference value of 1025. It is useful for coarsegraining
        # And lateral operators
        rho0 = 1025.
        wet = self.param.wet
        return wet * rho_array + (1-wet) * rho0
        
    @property
    def Nsquared(self):
        '''
        Here we compute the square of Brunt-Vaisala frequency (1/s^2) according to 
        formula (https://mom-ocean.github.io/assets/pdfs/MOM5_manual.pdf), Eq. (23.274):
        N^2 = - g/rho0 * d rho / dz,
        and parameters are given by Elizabeth Yankovsky
        where g = 9.8 m/s^2 is the gravitational acceleration
        rho0 = 1025 kg / m^3 is the reference density
        rho is the POTENTIAL density (MOM5 manual and https://en.wikipedia.org/wiki/Brunt%E2%80%93V%C3%A4is%C3%A4l%C3%A4_frequency),
        also Vallis Book Eq. (2.243), Shelton 1998 Appendix B.a.

        The result is defined on the vertical interfaces between finite-volume cells
        '''

        rho = self.rho()
        grid = self.grid
        param = self.param
        # Minus is not needed here because 'Z' is directed downward
        # Also note that drho/dz equals zero on the surface and on the bottom (same as wet_w),
        # also it is defined on cell interfaces, which are in midpoints w.r.t. zl position
        # Integration factor for center point (boundary values do not matter because of masking)
        dzB = grid.diff(param.zl,'Z')
        drho_dz = grid.diff(param.wet * rho.chunk({'zl': -1}),'Z') / dzB * param.wet_w
        
        g = 9.8
        rho0 = 1025.

        return np.maximum(g * drho_dz / rho0, 0.)

    @property
    def baroclinic_speed(self):
        '''
        Baroclinic Gravity Wave Speed,
        see https://mom-ocean.github.io/assets/pdfs/MOM5_manual.pdf, Eq. (23.276)
        c = 1/pi * integral(N, dz)

        The output is in m/s
        '''
        N = np.sqrt(self.Nsquared)

        grid = self.grid
        param = self.param
        
        # Integration factors for each interface is
        # the difference between center points
        # Also, no integration at the surface and at the bottom
        dzB = grid.diff(param.zl, 'Z')
        dzB[0] = 0; dzB[-1] = 0
        
        return 1./np.pi * (N * dzB).sum('zi')

    @property
    def deformation_radius(self):
        '''
        According to https://mom-ocean.github.io/assets/pdfs/MOM5_manual.pdf, 
        we consider two definitions of Rossby deformation radius:
        Rd = cg / |f| for large latitudes
        and 
        Rd = sqrt(cg / (2 * beta)) for low latitudes, where
        "cg" is the  Baroclinic Gravity Wave Speed
        f is the Coriolis parameter

        Following MOM6 code https://github.com/Pperezhogin/MOM6/blob/dev/gfdl/src/parameterizations/lateral/MOM_lateral_mixing_coeffs.F90#L269-L270, 
        and Hallberg 2013 https://www.sciencedirect.com/science/article/pii/S1463500313001601
        we combine both definitions of the deformation radius into single expression:
        Rd = cg / sqrt(f^2 + cg * 2 * beta)

        The output of deformation radius (Rd) is in m
        '''

        f, beta = Coriolis(self.param.yh, compute_beta=True)

        cg = self.baroclinic_speed

        Rd = cg / np.sqrt(f**2 + cg * 2 * beta)

        return Rd

    @property
    def Rd_dx(self):
        '''
        Deformation radius divided by the grid spacing.
        We keep definition of grid spacing similar to MOM6 as (dx^2+dy^2)^1/2
        https://github.com/Pperezhogin/MOM6/blob/dev/gfdl/src/parameterizations/lateral/MOM_lateral_mixing_coeffs.F90#L1546
        '''

        param = self.param
        dx = np.sqrt(param.dxT**2 + param.dyT**2)
        return self.deformation_radius / dx

    @property
    def rossby_number(self):
        '''
        Local Rossby number is defined similarly to Klower 2017, Juricke 2019:
        as a local 2D or 3D field
        Ro = sqrt(D^2+D_hat^2) / |f|

        Possible usage is the resolution function:
        Resolution_function = 1/(1+Ro)
        '''
        sh_xy, sh_xx, vort_xy, div = self.velocity_gradients()

        Shear_mag = self.param.wet * (sh_xx**2+self.grid.interp(sh_xy**2,['X','Y']))**0.5

        f = Coriolis(self.param.yh)
        Ro = Shear_mag / (np.abs(f)+1e-25)
        return Ro
    
    def vertical_shear(self):
        '''
        Compute vertical shear from cm2.6 data directly.
        Return the output in W points. 
        '''
        grid = self.grid
        param = self.param
        data = self.data

        u = grid.interp(data.u, 'X') * param.wet
        v = grid.interp(data.v, 'Y') * param.wet

        dzB = grid.diff(param.zl,'Z')
        uz = (grid.diff(u.chunk({'zl':-1}),'Z') / dzB) * param.wet_w
        vz = (grid.diff(v.chunk({'zl':-1}),'Z') / dzB) * param.wet_w

        return uz, vz
    
    def vertical_shear_geostrophic(self, potential=False):
        '''
        Compute vertical shear for the geostrophically
        balanced motion only assuming thermal wind balance.
        The default algorithm uses in-situ density.
        du/dz = - g/(rho0 * f) * drho/dy
        dv/dz = + g/(rho0 * f) * drho/dx
        '''
        grid = self.grid
        param = self.param

        g = 9.8
        rho0 = 1025.
        f = Coriolis(param.yh)

        rho = self.rho(potential=potential) # in-situ or potential density
        uz  = - g /(rho0 * f) * grid.interp(grid.diff(rho.chunk({'zl':-1}),  'Y') / param.dyCv * param.wet_v, ['Y','Z']) * param.wet_w
        vz  = + g /(rho0 * f) * grid.interp(grid.diff(rho.chunk({'zl':-1}),  'X') / param.dxCu * param.wet_u, ['X','Z']) * param.wet_w

        return uz, vz
    
    def Eady_time(self, potential_density_to_compute_vertical_shear=False, 
                    depth_threshold=0.):
        '''
        Following Smith 2007 "The geography of linear baroclinic instability in Earth’s oceans"
        and Held and Larichev 1996, we introduce the Eady time Te as follows:
        1/Te = f sqrt(average(1/Ri(z))), where 
        Ri = N^2 / (U_z^2 + V_z^2)

        The following numerical features are introduced to ease the computation:
        * We use geostrophic vertical shear instead of vertical shear in model output, thus resulting to:
        1/Te = sqrt(g/rho0 * average((rho_x^2 + rho_y^2) / rho_z))
        * We exclude points with negative N^2 from depth-averaging
        * It is best to use this function when coarsegraining is performed with percentile=1.
        '''
        g = 9.8
        rho0 = 1025.
        grid = self.grid
        param = self.param

        rhop = self.rho(potential=True)
        if potential_density_to_compute_vertical_shear:
            rho = rhop
        else:
            rho = self.rho(potential=False)

        dzB = grid.diff(param.zl,'Z')

        # We exlude placing B.C. up until the end
        drho_dz = grid.diff(param.wet * rhop.chunk({'zl': -1}),'Z') / dzB
        drho_dy = grid.interp(grid.diff(rho.chunk({'zl':-1}),  'Y') / param.dyCv * param.wet_v, ['Y','Z'])
        drho_dx = grid.interp(grid.diff(rho.chunk({'zl':-1}),  'X') / param.dxCu * param.wet_u, ['X','Z'])

        rho_nabla2 = drho_dx**2 + drho_dy**2

        # Here we form mask of where to average
        mask = np.logical_and(drho_dz>0, param.wet_w==1)
        # Add depth threshold if specified
        if depth_threshold > 0.:
            mask = np.logical_and(mask, param.zi>depth_threshold)

        integrated = rho_nabla2 / drho_dz
        weights = xr.where(mask, dzB, 0)

        invTe = np.sqrt(g/rho0 * (integrated * weights).sum('zi') / weights.sum('zi'))

        # Eady time
        Te = 1. / invTe

        # Inverse Richardson
        f = Coriolis(param.yh)
        invRi = (invTe/f)**2

        # Diagnostic output
        invRi_local = integrated * g / (rho0 * f**2)

        return Te, invRi, invRi_local

    def Eady_time_direct(self, N2_small=1e-8):
        '''
        Following Smith 2007 "The geography of linear baroclinic instability in Earth’s oceans",
        we introduce Richardson number as follows:
        Ri = N^2 / (U_z^2 + V_z^2), and then compute the parameter for water column:
        1/H int(1/Ri(z), z=-H..0)

        N2_small=1e-8s-2 following CHelton 1998
        '''
        N2_small=1e-8

        N2 = self.Nsquared

        grid = self.grid
        param = self.param
        data = self.data

        # Define vertical velocities on the interfaces
        # On the surface and bottom derivatives are zero (same as Nsquared)
        u = grid.interp(data.u, 'X') * param.wet
        v = grid.interp(data.v, 'Y') * param.wet

        dzB = grid.diff(param.zl,'Z')
        u_z = grid.diff(u.chunk({'zl':-1}),'Z') / dzB
        v_z = grid.diff(v.chunk({'zl':-1}),'Z') / dzB

        # multiplying by mask eliminates surface and bottom contribution
        invRi_local = ((u_z**2 + v_z**2) / (N2+N2_small)) * param.wet_w

        # Doint integration
        numerator = invRi_local * dzB
        denominator = dzB * param.wet_w

        invRi = numerator.sum('zi') / denominator.sum('zi')
        f = Coriolis(param.yh)
        invTe = np.abs(f) * np.sqrt(invRi)
        Te = 1 / invTe
    
        return Te, invRi, invRi_local
    
    def Eady_time_simple(self, depth_threshold=-1.):
        '''
        Here we estimate Eady time scale simply as:
        Te = Ld / delta(U),
        where Ld is the Rossby deformation radius, and
        delta(U) = int(|u_z|, dz) = sum(|np.diff(u)|)
        '''
        
        grid = self.grid
        param = self.param
        data = self.data

        u = grid.interp(data.u, 'X') * param.wet
        v = grid.interp(data.v, 'Y') * param.wet

        # Just finite differences, because we will eventually integrate
        u_z = grid.diff(u.chunk({'zl':-1}),'Z')
        v_z = grid.diff(v.chunk({'zl':-1}),'Z')

        Uz = np.sqrt(u_z**2 + v_z**2)

        deltaU = (Uz * param.wet_w * (param.zi>depth_threshold)).sum('zi')
        Ld = self.deformation_radius
        Te = Ld / deltaU
        return Te, Ld, deltaU
    
    def baroclinic_velocities(self):
        '''
        Here we estimate denominator for 
        Eady time scale (Te = Ld / delta(U)) simply as:
        delta(U) is the scale of baroclinic velocities
        given by square root of twice the baroclinic kinetic energy
        where Ld is the Rossby deformation radius, and
        '''
        
        grid = self.grid
        param = self.param
        data = self.data

        u = grid.interp(data.u, 'X')
        v = grid.interp(data.v, 'Y')

        dzT = grid.diff(param.zi, 'Z') * param.wet
        mean_z = lambda x: (x * dzT).sum('zl') / dzT.sum('zl')

        u_mean = mean_z(u)
        v_mean = mean_z(v)
        u2_mean = mean_z((u)**2)
        v2_mean = mean_z((v)**2)

        # Here we assume that mean(u^2)-mean(u)**2 = mean((u-mean(u))^2)
        deltaU = np.sqrt(u2_mean + v2_mean - u_mean**2 - v_mean**2)

        return deltaU
    
    def prepare_features(self):
        '''
        This function prepares input features for physics-aware
        neural network. The following features are collected:
        * Rd, Rossby deformation radius, 2D map
        * Te, Eady time scale, 2D map
        * z_s = int(N(z'),z'=-z..0) / int(N(z'),z'=-H..0),
            3D map of non-dimensional vertical coordinate

        Note: This function to be called for a single time moment
        Note: This function repeats calculations of
              def deformation_radius to safe computations
        '''
        grid = self.grid
        param = self.param
        data = self.data

        # Compute N (buoyancy frequency)
        # Note that Nsquared is already zero at bottom
        N = np.sqrt(self.Nsquared)
        # Integration factor
        dzB = grid.diff(param.zl, 'Z')
        dzB[0] = 0; dzB[-1] = 0

        # Integral int(N(z'),z'=-z..0)
        Ndz = (N * dzB).cumsum('zi')
        # Full integral
        NH = Ndz[-1] 
        # Non-dimensional vertical coordinate
        # Set coordinate to zero in points where stratification is
        # inverse and hence the integral is zero
        z_s = xr.where(NH > 0, Ndz / NH, 0.)
        # The coordrinate is defined in center cells:
        # with zero at first grid point
        z_s = z_s.isel(zi=slice(0,-1)).rename({'zi': 'zl'})
        z_s['zl'] = self.data.zl

        # Baroclinic speed
        # Note: The last element of cumsum is sum
        cg = 1/np.pi * NH

        # Deformation radius
        f, beta = Coriolis(self.param.yh, compute_beta=True)
        Rd = cg / np.sqrt(f**2 + cg * 2 * beta)

        # Eady time scale
        deltaU = self.baroclinic_velocities()
        # If vertical shear is zero, the Eady time is infinity
        # However, we set it to most probable value which is
        # 10 days (10. * 86400s)
        Te = xr.where(deltaU>0, Rd / deltaU, 10.*86400)

        data['deformation_radius'] = Rd
        data['eady_time'] = Te
        data['rescaled_depth'] = z_s
        return data

    def vertical_modes(self, lon=0, lat=0, time=0, N2_small=1e-8,
        dirichlet_surface=False, dirichlet_bottom=False, few_modes=1):
        '''
        Wrapper for function vertical_modes working as follows:
        * Select data for one water column
        * Find bottom by looking mask
        * Compute N squared
        * Compute grid steps
        * Prepare data in wet points
        * Call vertical_modes_one_column
        * Wrap output to the xarray: modes and corresponding 
          internal gravity wave velocities
        '''

        grid = self.grid
        param = self.param
        
        # Check that there are points to compute the EBT structure
        wet = param.wet.sel(xh=lon, yh=lat, method='nearest')
        if wet[0] == 0.:
            print('Error: There is no water here. Check different point.')
            return

        # Count the number of wet finite volumes
        Zl = np.sum(wet.values.astype('int'))

        # Computing N squared for a given watercolumn
        N2 = self.Nsquared.sel(xh=lon, yh=lat, method='nearest').isel(time=time).values

        # Creating grid steps
        dzB = grid.diff(param.zl, 'Z')
        dzT = grid.diff(param.zi, 'Z')
        
        # Only internal interfaces: exclude surface and bottom
        N2_np = np.array(N2[1:Zl])
        dzB_np = np.array(dzB[1:Zl])

        # Exclude bottom: only wet points
        dzT_np = np.array(dzT[0:Zl])

        # For a while, I return here data which is needed 
        # to call the numpy function
        return N2_np, dzB_np, dzT_np, -param.zi[1:Zl], -param.zl[0:Zl]

        modes, cg = vertical_modes_one_column(N2_np, dzB_np, dzT_np, N2_small=N2_small, 
                                                dirichlet_surface=dirichlet_surface, 
                                                dirichlet_bottom=dirichlet_bottom,
                                                debug=False, few_modes=few_modes)

        modes = xr.DataArray(modes, dims=['zl', 'mode'], coords={'zl': param.zl[0:Zl]})
        return modes, cg

    def vertical_modes_map(self, time=0, N2_small=1e-8,
        dirichlet_surface=False, dirichlet_bottom=False, few_modes=1):
        '''
        This is extension of the function above which computes
        baroclinic modes for the full 3D snapshot
        '''

        grid = self.grid
        param = self.param

        # Computing N squared
        N2 = self.Nsquared.isel(time=time).compute().values

        # Creating grid steps
        dzB = grid.diff(param.zl, 'Z').values
        dzT = grid.diff(param.zi, 'Z').values

        nx, ny, nz = len(param.xh), len(param.yh), len(param.zl)

        Modes = np.zeros([nz, ny, nx, few_modes])
        Cg = np.zeros([ny, nx, few_modes])
        
        for j in range(ny):
            print(j)
            for i in range(nx):
                # Check that there are points to compute the EBT structure
                wet = param.wet.isel(xh=i, yh=j)

                # Count the number of wet finite volumes
                Zl = np.sum(wet.values.astype('int'))

                # Skip one-layer and less water columns
                if Zl <= 1:
                    continue
                
                # Only internal interfaces: exclude surface and bottom
                N2_np = N2[1:Zl,j,i]
                dzB_np = dzB[1:Zl]

                # Exclude bottom: only wet points
                dzT_np = dzT[0:Zl]

                modes, cg = vertical_modes_one_column(N2_np, dzB_np, dzT_np, N2_small=N2_small, 
                                                        dirichlet_surface=dirichlet_surface, 
                                                        dirichlet_bottom=dirichlet_bottom,
                                                        debug=False, few_modes=few_modes)

                Modes[:Zl,j,i,:] = modes
                Cg[j,i,:] = cg

        Modes = xr.DataArray(Modes, dims=['zl', 'yh', 'xh', 'mode'], 
                coords={'xh': param.xh, 'yh': param.yh, 'zl': param.zl})
        Cg = xr.DataArray(Cg, dims=['yh', 'xh', 'mode'], 
                coords={'xh': param.xh, 'yh': param.yh})

        return Modes, Cg