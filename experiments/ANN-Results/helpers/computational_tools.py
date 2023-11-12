from xgcm import Grid

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
    x = select_LatLon(variable, Lat=(15, 89), Lon=(-90,-10))
    if 'time' in x.dims:
        if time is None:
            x = x.isel(time=-1)
        else:
            x = x.isel(time=time)
    return x

class Experiment():
    def __init__(self,data, param):
        self.data = data
        self.param = param
        self.grid = Grid(param, coords={
                    'X': {'center': 'xh', 'right': 'xq'},
                    'Y': {'center': 'yh', 'right': 'yq'}
                    },
                    boundary={'X':'periodic', 'Y':'fill'},
                    fill_value = {'X':0, 'Y':0}
                    )
        
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