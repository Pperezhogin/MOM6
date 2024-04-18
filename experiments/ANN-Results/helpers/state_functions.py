import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import torch
import xrft
import gsw
from xgcm.padding import pad as xgcm_pad
from functools import lru_cache

from helpers.ann_tools import image_to_3x3_stencil_gpt, import_ANN, torch_pad, tensor_from_xarray
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

    def Apply_ANN(self, ann_Txy=None, ann_Txx_Tyy=None, time_revers=False, rotation=0, reflect_x=False, reflect_y=False):
        '''
        The only input is the dataset itself.
        The output is predicted momentum flux in physical
        units in torch format, and its divergence
        '''
        if 'time' in self.data.dims:
            raise NotImplementedError("This operation is not implemented for many time slices. Use a single time.")
        if ann_Txy is None:
            ann_Txy = import_ANN('trained_models/ANN_Txy_ZB.nc')
            print('Warning: Prediction from default ANN')
        if ann_Txx_Tyy is None:
            ann_Txx_Tyy = import_ANN('trained_models/ANN_Txx_Tyy_ZB.nc')
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
            Norm is computed with double      ision to prevent overflow
            '''
            return torch.sqrt((x.type(torch.float64)**2).sum(dim=-1, keepdims=True)).type(torch.float32)

        def extract_3x3(x):
            y = torch_pad(x, left=True, right=True, top=True, bottom=True)
            return image_to_3x3_stencil_gpt(y, 
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
                        extract_3x3(sh_xy   * (rotation_sign * reflect_sign * reverse_sign)), 
                        extract_3x3(sh_xx_q * (rotation_sign * reverse_sign)), 
                        extract_3x3(vort_xy * (reflect_sign  * reverse_sign))
                        ],-1)

        # Normalize input features
        input_norm = norm(input_features)
        input_features = (input_features / (input_norm+1e-30))

        # Make prediction with transforming prediction back to original frame
        Txy = ann_Txy(input_features) * (rotation_sign * reflect_sign * reverse_sign)

        # Now denormalize the output
        Txy = - Txy * input_norm * input_norm * (areaBu * wet_c).reshape(-1,1)
        Txy = Txy.reshape(sh_xy.shape)

        ########## Second, prediction of Txx, Tyy ###############
        input_features = torch.concat(
                        [
                        extract_3x3(sh_xy_h   * (rotation_sign * reflect_sign * reverse_sign)), 
                        extract_3x3(sh_xx     * (rotation_sign * reverse_sign)), 
                        extract_3x3(vort_xy_h * (reflect_sign  * reverse_sign))
                        ],-1)

        # Normalize input features
        input_norm = norm(input_features)
        input_features = (input_features / (input_norm+1e-30))

        # Make prediction
        Tdiag = ann_Txx_Tyy(input_features) * reverse_sign

        # Now denormalize the output
        Tdiag = - Tdiag * input_norm * input_norm * (areaT * wet).reshape(-1,1)
        
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
        
        Txx = Tdiag[:,Txx_idx].reshape(sh_xx.shape)
        Tyy = Tdiag[:,Tyy_idx].reshape(sh_xx.shape)
        
        Txx_padded = torch_pad(Txx * dyT**2, right=True)
        Txy_padded = torch_pad(Txy * dxBu**2, bottom=True)
        ZB20u = wet_u * (torch.diff(Txx_padded,dim=-1) / dyCu + torch.diff(Txy_padded,dim=-2) / dxCu) / (areaCu)
        
        Txy_padded = torch_pad(Txy * dyBu**2,left=True)
        Tyy_padded = torch_pad(Tyy * dxT**2, top=True)
        ZB20v = wet_v * (torch.diff(Txy_padded,dim=-1) / dyCv + torch.diff(Tyy_padded,dim=-2) / dxCv) / (areaCv)
        
        return {'Txx': Txx, 'Tyy': Tyy, 'Txy': Txy, 
                'ZB20u': ZB20u, 'ZB20v': ZB20v, 
                'sh_xx': sh_xx, 'sh_xy': sh_xy, 'vort_xy': vort_xy}
    
    def ANN(self, ann_Txy=None, ann_Txx_Tyy=None, time_revers=False, rotation=0, reflect_x=False, reflect_y=False):            
        pred = self.Apply_ANN(ann_Txy, ann_Txx_Tyy, time_revers, rotation, reflect_x, reflect_y)
        
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

    @property
    def rho(self):
        '''
        Compute POTENTIAL density in kg/m^3, which is sigma0+1000kg/m^3.
        We compute sigma0 according to https://www.teos-10.org/pubs/gsw/html/gsw_sigma0.html
        Note that instead of Absolute Salinity and Conservative Temperature we use
        standard CM2.6 variables: Potential temperature, degrees C and Practical Salinity, psu
        '''
        return (1000.+gsw.sigma0(self.data.salt, self.data.temp))
    
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

        rho = self.rho
        grid = self.grid
        param = self.param
        # Minus is not needed here because 'Z' is directed downward
        # Also note that drho/dz equals zero on the surface and on the bottom (same as wet_w),
        # also it is defined on cell interfaces, which are in midpoints w.r.t. zl position
        # Integration factor for center point (boundary values do not matter because of masking)
        dzT = grid.diff(param.zl,'Z')
        drho_dz = grid.diff(param.wet * rho.chunk({'zl': -1}),'Z') / dzT * param.wet_w
        
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
        dzC = grid.diff(param.zl, 'Z')
        dzC[0] = 0; dzC[-1] = 0
        
        return 1./np.pi * (N * dzC).sum('zi')

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

    # def EBT_mode(self, lon=0, lat=0, time=0, N2_small=1e-20):
    #     '''
    #     Calculates the Equivalent Barotropic Mode (EBT)
    #     following Yankovsky 2023 https://essopenarchive.org/doi/full/10.22541/essoar.169945317.74775304
    #     as follows:
    #     d/dz(f^2/N^2 dphi/dz) + lambda^2 phi = 0
    #     with boundary conditions dphi/dz=0 at surface
    #     and phi=0 at the bottom
    #     '''
    #     # Check that there are point to compute the EBT structure
    #     wet = self.param.wet.sel(xh=lon, yh=lat, method='nearest')
    #     if wet[0] == 0.:
    #         print('Error: There is no water here. Check different point.')
    #         return
        
    #     # This function shows when the first zero in mask appears.
    #     # Consequently, it is the number of wet grid points (in center of cells)
    #     # which are wet and where the eigenproblem will be solved
    #     Zl = np.argmin(wet)

    #     f = Coriolis(self.param.yh)
    #     f2 = float(f.sel(xh=lon, yh=lat, method='nearest')**2)

    #     # N2 is defined on interfaces ("zi")
    #     # We select lon,lat,time moment and bound N squared from below
    #     N2 = self.Nsquared.sel(xh=lon, yh=lat, method='nearest').isel(time=time).values + N2_small
        
    #     # For Zl center cells, we consider N^2 only on the interfaces between
    #     # plus two boundary interfaces values where are zero and will be ignored
    #     N2 = N2[0:Zl+1]
    #     N2inv = 1./N2
    #     # These values will be ignored in FD formulas
    #     N2inv[0] = 0; N2inv[-1] = 0

    #     # Grid coordinates of interfaces:
    #     zi = self.param.zi[0:Zl+1].values
    #     # Grid coordinates of centers
    #     zl = self.param.zl[0:Zl]
        
    #     diagonal = - f2 * (N2inv[0:Zl])




    #     # We solve the eigenvalue problem in the center of the cells









        




