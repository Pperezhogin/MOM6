from xgcm import Grid
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from helpers.ann_tools import image_to_3x3_stencil_gpt, import_ANN
import torch
from xgcm.padding import pad
import cartopy.crs as ccrs
import cmocean
import xrft

roundoff = 1e-40

def x_coord(array):
    '''
    Returns horizontal coordinate, 'xq' or 'xh'
    as xarray
    '''
    try:
        coord = array.xq
    except:
        coord = array.xh
    return coord

def y_coord(array):
    '''
    Returns horizontal coordinate, 'yq' or 'yh'
    as xarray
    '''
    try:
        coord = array.yq
    except:
        coord = array.yh
    return coord

def select_LatLon(array, Lat=(35,45), Lon=(5,15)):
    '''
    array is xarray
    Lat, Lon = tuples of floats
    '''
    x = x_coord(array)
    y = y_coord(array)

    return array.sel({x.name: slice(Lon[0],Lon[1]), 
                      y.name: slice(Lat[0],Lat[1])})

def select_NA(variable, time=None):
    x = select_LatLon(variable, Lat=(15, 65), Lon=(-90,-10))
    if 'time' in x.dims:
        if time is None:
            x = x.isel(time=-1)
        else:
            x = x.isel(time=time)
    return x

def select_Pacific(variable, time=None):
    x = select_LatLon(variable, Lat=(15, 65), Lon=(-250,-120))
    if 'time' in x.dims:
        if time is None:
            x = x.isel(time=-1)
        else:
            x = x.isel(time=time)
    return x

def select_Equator(variable, time=None):
    x = select_LatLon(variable, Lat=(-20, 20), Lon=(-230,-90))
    if 'time' in x.dims:
        if time is None:
            x = x.isel(time=-1)
        else:
            x = x.isel(time=time)
    return x

def select_Cem(variable, time=None):
    x = select_LatLon(variable, Lat=(-10,15), Lon=(-260,-230))
    if 'time' in x.dims:
        if time is None:
            x = x.isel(time=-1)
        else:
            x = x.isel(time=time)
    return x

# We compare masked fields because outside there may be 1e+20 values
def compare(tested, control, mask=None, vmax=None, selector=select_NA):
    if mask is not None:
        mask_nan = mask.data.copy()
        mask_nan[mask_nan==0.] = np.nan
        tested = tested * mask_nan
        control = control * mask_nan
    tested = selector(tested).compute()
    control = selector(control).compute()
    
    if vmax is None:
        vmax = control.std() * 2
    
    central_latitude = float(y_coord(control).mean())
    central_longitude = float(x_coord(control).mean())
    fig, axes = plt.subplots(2,2, figsize=(12, 10), subplot_kw={'projection': ccrs.Orthographic(central_latitude=central_latitude, central_longitude=central_longitude)})
    cmap = cmocean.cm.balance
    cmap.set_bad('gray')
    
    ax = axes[0][0]; ax.coastlines(); gl = ax.gridlines(); gl.bottom_labels=True; gl.left_labels=True;
    im = tested.plot(ax=ax, vmax=vmax, transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=False)
    ax.set_title('Tested field')
    ax = axes[0][1]; ax.coastlines(); gl = ax.gridlines(); gl.bottom_labels=True; gl.left_labels=True;
    control.plot(ax=ax, vmax=vmax, transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=False)
    ax.set_title('Control field')
    ax = axes[1][0]; ax.coastlines(); gl = ax.gridlines(); gl.bottom_labels=True; gl.left_labels=True;
    (tested-control).plot(ax=ax, vmax=vmax, transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=False)
    ax.set_title('Tested-control')
    plt.tight_layout()
    plt.colorbar(im, ax=axes, shrink=0.9, aspect=30, extend='both')
    axes[1][1].remove()
    
    ########## Metrics ##############
    error = tested-control
    relative_error = np.abs(error).mean() / np.abs(control).mean()
    R2 = 1 - (error**2).mean() / (control**2).mean()
    optimal_scaling = (tested*control).mean() / (tested**2).mean()
    error = tested * optimal_scaling - control
    R2_max = 1 - (error**2).mean() / (control**2).mean()
    corr = xr.corr(tested, control)
    print('Correlation:', float(corr))
    print('Relative Error:', float(relative_error))
    print('R2 = ', float(R2))
    print('R2 max = ', float(R2_max))
    print('Optinal scaling:', float(optimal_scaling))
    print(f'Nans [test/control]: [{int(np.sum(np.isnan(tested)))}, {int(np.sum(np.isnan(control)))}]')

class StateFunctions():
    def __init__(self, data, param, grid):
        self.data = data
        self.param = param
        self.grid = grid
        
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
        dEdt = (grid.interp(Ex * areaU,'X') + grid.interp(Ey * areaV,'Y')) * param.wet
        
        # Energy source Galilean-invariant form
        if Txy is not None:
            sh_xy, sh_xx, vort_xy = self.velocity_gradients()
            dEdt_G = - 0.5 * ((Txx * sh_xx - Tyy * sh_xx) * areaT + \
                grid.interp(2 * Txy * sh_xy * areaB,['X','Y']))
        else:
            dEdt_G = None

        # Enstrophy source
        f = self.relative_vorticity(fx,fy)
        vorticity = self.relative_vorticity(data.u,data.v)
        dZdt = f * vorticity * areaB
        
        # Palinstrophy source
        def gradient(w):
            wx = grid.diff(w, 'X') / param.dxCv * param.wet_v
            wy = grid.diff(w, 'Y') / param.dyCu * param.wet_u
            return wx, wy
        
        px, py = gradient(f)
        wx, wy = gradient(vorticity)
        Px = px * wx * areaV
        Py = py * wy * areaU
        dPdt = (grid.interp(Px,'Y') + grid.interp(Py,'X')) * param.wet

        return {'dEdt_G': dEdt_G, 'dEdt_local': dEdt, 'dZdt_local': dZdt, 'dPdt_local': dPdt, 'dEdt': float((dEdt).sum()), 'dZdt': float((dZdt).sum()), 'dPdt': float((dPdt.sum()))}    
           
    def transfer(self, fu_in, fv_in,
            region = 'NA', window='hann', 
            nfactor=2, truncate=False, detrend='linear', 
            window_correction=True, compensated=True):
        
        if region == 'NA':
            kw = {'Lat': (40,60), 'Lon': (-40,-20)}
        elif region == 'Pacific':
            kw = {'Lat': (20,40), 'Lon': (-200,-180)}
        else:
            print('Error: wrong region')
            
        # Select desired Lon-Lat square
        u = select_LatLon(self.data.u,**kw)
        v = select_LatLon(self.data.v,**kw)
        fu = select_LatLon(fu_in,**kw)
        fv = select_LatLon(fv_in,**kw)
        
        if u.shape != v.shape:
            shape = [None]*2
            shape[0] = min(u.shape[0], v.shape[0])
            shape[1] = min(u.shape[1], v.shape[1])
            def sel(x):
                return x[:shape[0],:shape[1]]
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

        Eu = xrft.isotropic_cross_spectrum(u, fu, dim=('xq','yh'), window=window, nfactor=nfactor, 
            truncate=truncate, detrend=detrend, window_correction=window_correction)
        Ev = xrft.isotropic_cross_spectrum(v, fv, dim=('xh','yq'), window=window, nfactor=nfactor, 
            truncate=truncate, detrend=detrend, window_correction=window_correction)
        
        E = (Eu+Ev)
        E['freq_r'] = E['freq_r']*2*np.pi # because library returns frequencies, but not wavenumbers

        if compensated:
            return np.real(E) * E['freq_r']
        else:
            return np.real(E)
    
    def transfer_ANN(self, ann_Txy, ann_Txx_Tyy, kw_ann={}, kw_sp={}):
        ann = self.ANN(ann_Txy, ann_Txx_Tyy, **kw_ann)
        return self.transfer(ann['ZB20u'], ann['ZB20v'], **kw_sp)
        
    def velocity_gradients(self):
        param = self.param
        data = self.data
        grid = self.grid
        
        dudx = grid.diff(data.u * param.wet_u / param.dyCu, 'X') * param.dyT / param.dxT * param.wet
        dvdy = grid.diff(data.v * param.wet_v / param.dxCv, 'Y') * param.dxT / param.dyT * param.wet

        dudy = grid.diff(data.u * param.wet_u / param.dxCu, 'Y') * param.dxBu / param.dyBu * param.wet_c
        dvdx = grid.diff(data.v * param.wet_v / param.dyCv, 'X') * param.dyBu / param.dxBu * param.wet_c
        
        sh_xx = dudx-dvdy
        sh_xy = dvdx+dudy
        vort_xy=dvdx-dudy
        
        return sh_xy, sh_xx, vort_xy
    
    def Smagorinsky(self, Cs_biharm=0.06):
        sh_xy, sh_xx, vort_xy = self.velocity_gradients()
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
        
    def ZB20(self, ZB_scaling=1.0):
        param = self.param
        grid = self.grid
        
        sh_xy, sh_xx, vort_xy = self.velocity_gradients()

        vort_xy_center = grid.interp(vort_xy,['X','Y']) * param.wet
        sh_xy_center = grid.interp(sh_xy,['X','Y']) * param.wet
        sh_xx_corner = grid.interp(sh_xx,['X', 'Y']) * param.wet_c

        vort_sh = vort_xy_center * sh_xy_center

        sum_sq = 0.5 * (vort_xy_center**2 + sh_xy_center**2 + sh_xx**2)

        kappa_t = - param.dxT * param.dyT * param.wet * ZB_scaling
        kappa_q = - param.dxBu * param.dyBu * param.wet_c * ZB_scaling


        Txx = kappa_t * (- vort_sh + sum_sq)
        Tyy = kappa_t * (+ vort_sh + sum_sq)
        Txy = kappa_q * (vort_xy * sh_xx_corner)

        ZB20u = param.wet_u * (grid.diff(Txx*param.dyT**2, 'X') / param.dyCu     \
               + grid.diff(Txy*param.dxBu**2, 'Y') / param.dxCu) \
               / (param.dxCu*param.dyCu)

        ZB20v = param.wet_v * (grid.diff(Txy*param.dyBu**2, 'X') / param.dyCv     \
                   + grid.diff(Tyy*param.dxT**2, 'Y') / param.dxCv) \
                   / (param.dxCv*param.dyCv)

        return {'ZB20u': ZB20u, 'ZB20v': ZB20v, 
                'Txx': Txx, 'Tyy': Tyy, 'Txy': Txy}
    
    def Apply_ANN(self, ann_Txy=None, ann_Txx_Tyy=None, time_revers=False, rotation=0, reflect_x=False, reflect_y=False):
        '''
        The only input is the dataset itself.
        The output is predicted momentum flux in physical
        units in torch format
        '''
        if 'time' in self.data.dims:
            raise NotImplementedError("This operation is not implemented for many time slices. Use a single time.")
        if ann_Txy is None:
            ann_Txy = import_ANN('trained_models/ANN_Txy_ZB.nc')
            print('Warning: Prediction from default ANN')
        if ann_Txx_Tyy is None:
            ann_Txx_Tyy = import_ANN('trained_models/ANN_Txx_Tyy_ZB.nc')
            print('Warning: Prediction from default ANN')
        
        def norm(x):
            return torch.sqrt((x**2).sum(dim=-1, keepdims=True))
        
        def tensor(x, torch_type=torch.float32):
            return torch.tensor(x.values).type(torch_type)

        grid = self.grid
        param = self.param

        def _pad(x):
            return pad(x, grid, {'X':1, 'Y':1})

        def extract_3x3(x):
            return image_to_3x3_stencil_gpt(tensor(_pad(x)), 
                rotation=rotation, reflect_x=reflect_x, reflect_y=reflect_y)
        
        sh_xy, sh_xx, vort_xy = self.velocity_gradients()

        sh_xy_h = grid.interp(sh_xy, ['X','Y']) * param.wet
        vort_xy_h = grid.interp(vort_xy, ['X','Y']) * param.wet
        sh_xx_q = grid.interp(sh_xx, ['X','Y']) * param.wet_c

        ########## First, do prediction for Txy stress ###########
        ## Note on rotation ##
        # Consider rotation=90 as an example
        # We need not only to rotate input features, but also
        # change sign of input sh_xy and sh_xx
        # Also, output Txy should flip sign, but Txx and Tyy should
        # flip each other
        # Vorticity is a scalar and it does not change sign
        if rotation in [0, 180]:
            rotation_sign = 1
        elif rotation in [90, 270]:
            rotation_sign = -1
        else:
            print('Error: use rotation one of 0, 90, 180, 270')
        
        ## Note on reflection ##
        # It does not change prediction of Txx, Tyy
        reflect_sign = 1
        if reflect_x:
            reflect_sign = - reflect_sign
        if reflect_y:
            reflect_sign = - reflect_sign
        
        # This key enables prediction of
        # -T(-features)
        # For Smagorinsky model, it is the same value
        # Reversibility does not contribute to feature rearrangement
        if time_revers:
            reverse_sign = -1
        else:
            reverse_sign = 1
        
        # Collect input features
        input_features = torch.concat(
                        [
                        extract_3x3(sh_xy   * (rotation_sign * reflect_sign * reverse_sign)), 
                        extract_3x3(sh_xx_q * (rotation_sign * reverse_sign)), 
                        extract_3x3(vort_xy * (reflect_sign  * reverse_sign))
                        ],-1)

        # Normalize input features
        input_norm = norm(input_features.type(torch.float64)).type(torch.float32)
        input_features = (input_features / (input_norm+1e-30))

        # Make prediction
        # Note: rotation sign here transforms prediction back
        # to original frame
        Txy = ann_Txy(input_features) * (rotation_sign * reflect_sign * reverse_sign)

        # Now denormalize the output
        area = tensor((param.dxBu * param.dyBu)).reshape(-1,1)
        Txy = - Txy * input_norm * input_norm * area
        Txy = Txy.reshape(sh_xy.shape)
        
        # Placing boundary conditions
        wet = tensor(param.wet_c)
        Txy = Txy * wet

        ########## Second, prediction of Txx, Tyy ###############
        input_features = torch.concat(
                        [
                        extract_3x3(sh_xy_h   * (rotation_sign * reflect_sign * reverse_sign)), 
                        extract_3x3(sh_xx     * (rotation_sign * reverse_sign)), 
                        extract_3x3(vort_xy_h * (reflect_sign  * reverse_sign))
                        ],-1)

        # Normalize input features
        input_norm = norm(input_features.type(torch.float64)).type(torch.float32)
        input_features = (input_features / (input_norm+1e-30))

        # Make prediction
        Tdiag = ann_Txx_Tyy(input_features) * reverse_sign

        # Now denormalize the output
        area = tensor((param.dxT * param.dyT)).reshape(-1,1)
        Tdiag = - Tdiag * input_norm * input_norm * area
        
        # This transformation transforms the prediction 
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
        
        # Placing boundary conditions
        wet = tensor(param.wet)
        Txx = Txx * wet
        Tyy = Tyy * wet
        
        wet_u = tensor(param.wet_u)
        wet_v = tensor(param.wet_v)
        dyT = tensor(param.dyT)
        dxT = tensor(param.dxT)
        dxCu = tensor(param.dxCu)
        dyCu = tensor(param.dyCu)
        dyCv = tensor(param.dyCv)
        dxCv = tensor(param.dxCv)
        dxBu = tensor(param.dxBu)
        dyBu = tensor(param.dyBu)
        
        from torch.nn.functional import pad as torch_pad
        
        def zonal_circular_pad(x, right=True):
            y = torch.zeros(x.shape[-2], x.shape[-1]+1)
            if right:
                y[:,:-1] = x
                y[:,-1] = x[:,0]
            else:
                y[:,1:] = x
                y[:,0] = x[:,-1]
            return y
                
        Txx_padded = zonal_circular_pad(Txx * dyT**2)
        Txy_padded = torch_pad(Txy * dxBu**2, (0,0,1,0)) # pad on the left with zero along meridional direction
        ZB20u = wet_u * (torch.diff(Txx_padded,dim=-1) / dyCu + torch.diff(Txy_padded,dim=-2) / dxCu) / (dxCu * dyCu)
        
        Txy_padded = zonal_circular_pad(Txy * dyBu**2,right=False)
        Tyy_padded = torch_pad(Tyy * dxT**2, (0,0,0,1)) # pad on the right with zero along meridional direction
        ZB20v = wet_v * (torch.diff(Txy_padded,dim=-1) / dyCv + torch.diff(Tyy_padded,dim=-2) / dxCv) / (dxCv * dyCv)
        
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
        dvdx = grid.diff(param.wet_v * v*dyCv,'X')
        dudy = grid.diff(param.wet_u * u*dxCu,'Y')
        return (dvdx - dudy) * IareaBu * param.wet_c

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