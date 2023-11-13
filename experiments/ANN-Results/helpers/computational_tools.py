from xgcm import Grid
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from helpers.ann_tools import image_to_3x3_stencil_gpt, import_ANN
import torch
from xgcm.padding import pad

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

# We compare masked fields because outside there may be 1e+20 values
def compare(tested, control, mask, vmax=None, selector=select_NA):
    tested = selector(tested * mask)
    control = selector(control * mask)
    
    plt.figure(figsize=(12,10))
    plt.subplot(2,2,1)
    tested.plot(vmax=vmax, robust=True)
    plt.xlabel(''); plt.ylabel('')
    plt.title('Tested field')
    plt.subplot(2,2,2)
    control.plot(vmax=vmax, robust=True)
    plt.title('Control field')
    plt.xlabel(''); plt.ylabel('')
    plt.subplot(2,2,3)
    np.abs((tested-control)).plot(vmax=vmax, robust=True)
    plt.title('Tested-control')
    plt.xlabel(''); plt.ylabel('')
    plt.tight_layout()
    error = tested-control
    relative_error = np.abs(error).mean() / np.abs(control).mean()
    R2 = 1 - (error**2).mean() / (control**2).mean()
    optimal_scaling = (tested*control).mean() / (tested**2).mean()
    corr = xr.corr(tested, control)
    print('Correlation:', float(corr))
    print('Relative Error:', float(relative_error))
    print('R2 = ', float(R2))
    print('Optinal scaling:', float(optimal_scaling))
    print('Nans:', int(np.sum(np.isnan(error))))

class StateFunctions():
    def __init__(self,data, param, grid):
        self.data = data
        self.param = param
        self.grid = grid
        
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

        ZB2020u = param.wet_u * (grid.diff(Txx*param.dyT**2, 'X') / param.dyCu     \
               + grid.diff(Txy*param.dxBu**2, 'Y') / param.dxCu) \
               / (param.dxCu*param.dyCu)

        ZB2020v = param.wet_v * (grid.diff(Txy*param.dyBu**2, 'X') / param.dyCv     \
                   + grid.diff(Tyy*param.dxT**2, 'Y') / param.dxCv) \
                   / (param.dxCv*param.dyCv)

        return {'ZB2020u': ZB2020u, 'ZB2020v': ZB2020v, 
                'Txx': Txx, 'Tyy': Tyy, 'Txy': Txy}
    
    def Apply_ANN(self, ann_Txy=None, ann_Txx_Tyy=None):
        '''
        The only input is the dataset itself.
        The output is predicted momentum flux in physical
        units in torch format
        '''
        
        def norm(x):
            return torch.sqrt((x**2).sum(dim=-1, keepdims=True))
        
        data = self.data
        grid = self.grid
        param = self.param

        def _pad(x):
            return pad(x, grid, {'X':1, 'Y':1})

        def extract_3x3(x):
            return image_to_3x3_stencil_gpt(torch.tensor(_pad(x).values))

        ########## First, do prediction for Txy stress ###########
        # Collect input features
        sh_xy = extract_3x3(data.sh_xy)
        sh_xx = extract_3x3(data.sh_xx_q)
        vort_xy = extract_3x3(data.vort_xy)
        input_features = torch.concat([sh_xy, sh_xx, vort_xy],-1).type(torch.float64)

        # Normalize input features
        input_norm = norm(input_features)
        input_features = (input_features / (input_norm+1e-70)).type(torch.float32)

        # Make prediction
        Txy = ann_Txy(input_features)

        # Now denormalize the output
        area = torch.tensor((param.dxBu * param.dyBu).values).reshape(-1,1)
        Txy = - Txy * input_norm * input_norm * area
        Txy = Txy.reshape(data.sh_xy.shape)

        ########## Second, prediction of Txx, Tyy ###############
        sh_xy = extract_3x3(data.sh_xy_h)
        sh_xx = extract_3x3(data.sh_xx)
        vort_xy = extract_3x3(data.vort_xy_h)
        input_features = torch.concat([sh_xy, sh_xx, vort_xy],-1).type(torch.float64)

        # Normalize input features
        input_norm = norm(input_features)
        input_features = (input_features / (input_norm+1e-70)).type(torch.float32)

        # Make prediction
        Tdiag = ann_Txx_Tyy(input_features)

        # Now denormalize the output
        area = torch.tensor((param.dxT * param.dyT).values).reshape(-1,1)
        Tdiag = - Tdiag * input_norm * input_norm * area
        Txx = Tdiag[:,0].reshape(data.sh_xx.shape)
        Tyy = Tdiag[:,1].reshape(data.sh_xx.shape)

        return Txy, Txx, Tyy
    
    def ANN(self, ann_Txy=None, ann_Txx_Tyy=None):
        if 'time' in self.data.dims:
            raise NotImplementedError("This operation is not implemented for many time slices. Use a single time.")
        if ann_Txy is None:
            ann_Txy = import_ANN('trained_models/ANN_Txy_ZB.nc')
        if ann_Txx_Tyy is None:
            ann_Txx_Tyy = import_ANN('trained_models/ANN_Txx_Tyy_ZB.nc')
            
        Txy, Txx, Tyy = self.Apply_ANN(ann_Txy, ann_Txx_Tyy)
        
        Txy = Txy.detach().numpy() + self.param.dxBu * 0
        Txx = Txx.detach().numpy() + self.param.dxT * 0
        Tyy = Tyy.detach().numpy() + self.param.dxT * 0

        return {'Txx': Txx, 'Tyy': Tyy, 'Txy': Txy}
    
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

    def relative_vorticity(self):
        '''
        https://github.com/NOAA-GFDL/MOM6/blob/dev/gfdl/src/core/MOM_CoriolisAdv.F90#L472
        '''
        param = self.param
        u = self.data.u
        v = self.data.v
        grid = self.grid
        
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