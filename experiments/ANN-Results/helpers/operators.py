import gcm_filters
import xarray as xr
import numpy as np
    
class Coarsen:
    def __call__(self, u=None, v=None, T=None, 
                 ds_hires=None, ds_coarse=None, factor=None):
        '''
        Algorithm: 
        * Interpolate velocities to the center
        * Coarsegrain
        * Interpolate to nodes of Arakawa-C grid on coarse grid

        Note: main reason for such algorithm is inability to coarsegrain
        exactly to side points of Arakawa-C grid
        Note: compared to direct coarsegraining and interpolation, 
        this algorithm is almost the same for large coarsegraining factor
        '''
        if factor ==1:
            return u, v, T
        
        coarsen = lambda x: x.coarsen({'xh':factor, 'yh':factor}).mean()
        u_coarse = None; v_coarse = None; T_coarse = None

        if u is not None:
            u_coarse = ds_coarse.grid.interp(
                    coarsen(ds_hires.grid.interp(u, 'X')*ds_hires.param.wet) \
                    * ds_coarse.param.wet,'X') * ds_coarse.param.wet_u
        if v is not None:
            v_coarse = ds_coarse.grid.interp(
                    coarsen(ds_hires.grid.interp(v, 'Y')*ds_hires.param.wet) \
                    * ds_coarse.param.wet,'Y') * ds_coarse.param.wet_v   
            
        if T is not None:
            T_coarse = coarsen(T) * ds_coarse.param.wet
            
        return u_coarse, v_coarse, T_coarse
    
class CoarsenWeighted:
    def __call__(self, u=None, v=None, T=None, 
                 ds_hires=None, ds_coarse=None, factor=None):
        '''
        Algorithm: 
        * Interpolate velocities to the center
        * Coarsegrain
        * Interpolate to nodes of Arakawa-C grid on coarse grid

        Note: we weight here all operations with local grid area
        '''

        coarsen = lambda x: x.coarsen({'xh':factor, 'yh':factor}).sum()
        u_coarse = None; v_coarse = None; T_coarse = None
        
        if u is not None:
            areaU = ds_hires.param.dxCu * ds_hires.param.dyCu
            u_weighted = ds_hires.grid.interp(u * areaU,'X') * ds_hires.param.wet
            
            areaU = ds_coarse.param.dxCu * ds_coarse.param.dyCu
            u_coarse = ds_coarse.grid.interp(
                coarsen(u_weighted) * ds_coarse.param.wet,'X') \
                * ds_coarse.param.wet_u / areaU

        if v is not None:
            areaV = ds_hires.param.dxCv * ds_hires.param.dyCv
            v_weighted = ds_hires.grid.interp(v * areaV,'Y') * ds_hires.param.wet
            
            areaV = ds_coarse.param.dxCv * ds_coarse.param.dyCv
            v_coarse = ds_coarse.grid.interp(
                coarsen(v_weighted) * ds_coarse.param.wet,'Y') \
                * ds_coarse.param.wet_v / areaV
    
        if T is not None:
            coarsen = lambda x: x.coarsen({'xh':factor, 'yh':factor}).sum()
            weights = xr.where(ds_hires.param.wet==1,
                               ds_hires.param.dxT * ds_hires.param.dyT,
                               np.nan)
            T_coarse = xr.where(ds_coarse.param.wet==1, 
                                coarsen(T * weights) / coarsen(weights),
                                0)
            
        return u_coarse, v_coarse, T_coarse
    
class CoarsenKochkov:
    def __call__(self, u=None, v=None, T=None, 
                 ds_hires=None, ds_coarse=None, factor=None):
        '''
        Algorithm: 
        * Apply weighted coarsegraining along cell side
        * Apply subsampling orthogonally to cell side
        
        Note: This coarsegraining allows to satisfy exactly the incompressibility
        and follows from finite-volume approach, Kochkov2021:
        https://www.pnas.org/doi/abs/10.1073/pnas.2101784118 (see their Supplementary)
        '''
        u_coarse = None; v_coarse = None; T_coarse = None

        if u is not None:
            u_coarse = (u * ds_hires.param.dyCu).coarsen({'yh': factor}).sum()[{'xq': slice(factor-1,None,factor)}] \
                    * ds_coarse.param.wet_u / ds_coarse.param.dyCu
        
        if v is not None:
            v_coarse = (v * ds_hires.param.dxCv).coarsen({'xh': factor}).sum()[{'yq': slice(factor-1,None,factor)}] \
                        * ds_coarse.param.wet_v / ds_coarse.param.dxCv
        
        if T is not None:
            coarsen = lambda x: x.coarsen({'xh':factor, 'yh':factor}).sum()
            areaT = ds_hires.param.dxT * ds_hires.param.dyT
            areaTc = ds_coarse.param.dxT * ds_coarse.param.dyT
            T_coarse = coarsen(T * areaT) * ds_coarse.param.wet / areaTc

        return u_coarse, v_coarse, T_coarse

class CoarsenKochkovMinMax:
    def __call__(self, u=None, v=None, T=None, 
                 ds_hires=None, ds_coarse=None, factor=None):
        '''
        This operator differs from CoarsenKochkov by different treatment
        of boundary conditions. Here B.C. are applied such that 
        min/max values are not violated.
        Algorithm:
        u_mean = sum(w_i u_i) / sum(w_i),
        where weights w_i contain nans. In that case they will
        be ignored by the sum operator
        '''
        u_coarse = None; v_coarse = None; T_coarse = None

        if u is not None:
            u_coarse = (u * ds_hires.param.dyCu).coarsen({'yh': factor}).sum()[{'xq': slice(factor-1,None,factor)}] \
                    * ds_coarse.param.wet_u / ds_coarse.param.dyCu
        
        if v is not None:
            v_coarse = (v * ds_hires.param.dxCv).coarsen({'xh': factor}).sum()[{'yq': slice(factor-1,None,factor)}] \
                        * ds_coarse.param.wet_v / ds_coarse.param.dxCv
        
        if T is not None:
            coarsen = lambda x: x.coarsen({'xh':factor, 'yh':factor}).sum()
            weights = xr.where(ds_hires.param.wet==1,
                               ds_hires.param.dxT * ds_hires.param.dyT,
                               np.nan)
            T_coarse = xr.where(ds_coarse.param.wet==1, 
                                coarsen(T * weights) / coarsen(weights),
                                0)

        return u_coarse, v_coarse, T_coarse

class Subsampling:
    def __call__(self, u=None, v=None, T=None, 
                 ds_hires=None, ds_coarse=None, factor=None):
        '''
        Algorithm: 
        * Apply interpolation along cell side
        * Apply subsampling orthogonally to cell side
        
        Note: This subsampling was used in 
        Xie 2020
        https://journals.aps.org/prfluids/abstract/10.1103/PhysRevFluids.5.054606,
        see below Eq. 17
        '''
        u_coarse = None; v_coarse = None; T_coarse = None

        if u is not None:
            u_coarse = u.interp(yh = ds_coarse.param.yh)[{'xq': slice(factor-1,None,factor)}] * ds_coarse.param.wet_u

        if v is not None:
            v_coarse = v.interp(xh = ds_coarse.param.xh)[{'yq': slice(factor-1,None,factor)}] * ds_coarse.param.wet_v
        
        if T is not None:
            T_coarse = T.interp(xh = ds_coarse.param.xh, yh = ds_coarse.param.yh)
        
        return u_coarse, v_coarse, T_coarse

class Filtering:
    def __init__(self, shape=gcm_filters.FilterShape.GAUSSIAN):
        super().__init__()
        self.shape = shape
    def __call__(self, u=None, v=None, T=None, 
                 ds_hires=None, 
                 FGR=2):
        '''
        Algorithm:
        * Initialize GCM-filters with a given FGR,
        informing with local cell wet mask
        * Apply filter without coarsegraining or subsampling
        '''
        u_filtered = None; v_filtered = None; T_filtered = None
        
        if u is not None:
            filter_simple_fixed_factor = gcm_filters.Filter(
                filter_scale=FGR,
                dx_min=1,
                filter_shape=self.shape,
                grid_type=gcm_filters.GridType.REGULAR_WITH_LAND,
                grid_vars={'wet_mask': ds_hires.param.wet_u}
                )
            u_filtered = filter_simple_fixed_factor.apply(u, dims=['yh', 'xq'])

        if v is not None:
            filter_simple_fixed_factor = gcm_filters.Filter(
                filter_scale=FGR,
                dx_min=1,
                filter_shape=self.shape,
                grid_type=gcm_filters.GridType.REGULAR_WITH_LAND,
                grid_vars={'wet_mask': ds_hires.param.wet_v}
                )
            v_filtered = filter_simple_fixed_factor.apply(v, dims=['yq', 'xh'])

        if T is not None:
            filter_simple_fixed_factor = gcm_filters.Filter(
                filter_scale=FGR,
                dx_min=1,
                filter_shape=self.shape,
                grid_type=gcm_filters.GridType.REGULAR_WITH_LAND,
                grid_vars={'wet_mask': ds_hires.param.wet}
                )
            T_filtered = filter_simple_fixed_factor.apply(T, dims=['yh', 'xh'])
            
        return u_filtered, v_filtered, T_filtered