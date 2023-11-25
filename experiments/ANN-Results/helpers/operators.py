import gcm_filters

class Operator():
    '''
    Operator for filtering and coarsegraining velocities on 
    staggered Arakawa-C grid
    '''
    def __init__(self):
        pass
    def __call__(self, u, v, ds_hires, ds_coarse):
        '''
        Assume:
        * u, v are on Arakawa-C grid
        * u, v are masked with wet mask
        * output is masked
        '''
        raise NotImplementedError
    
    def __add__(first, second):
        if not isinstance(second, Operator):
            raise TypeError("Unsupported operand type. The right operand must be an instance of Operator.")
        
        class CombinedOperator(Operator):
            def __init__(self):
                super().__init__()
                self.first = first
                self.second = second
            def __call__(self, u, v, ds_hires, ds_coarse):
                '''
                Here we sequentially apply the operators
                '''
                u_, v_ = self.first(u, v, ds_hires, ds_coarse)
                return self.second(u_, v_, ds_hires, ds_coarse)
        return CombinedOperator()
    
class Coarsen(Operator):
    def __init__(self):
        super().__init__()
    def __call__(self, u, v, ds_hires, ds_coarse):
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
        coarsen = lambda x: x.coarsen({'xh':ds_coarse.factor, 'yh':ds_coarse.factor}).mean()

        u_coarse = ds_coarse.grid.interp(
                coarsen(ds_hires.grid.interp(u, 'X')*ds_hires.param.wet) \
                * ds_coarse.param.wet,'X') * ds_coarse.param.wet_u
        
        v_coarse = ds_coarse.grid.interp(
                coarsen(ds_hires.grid.interp(v, 'Y')*ds_hires.param.wet) \
                * ds_coarse.param.wet,'Y') * ds_coarse.param.wet_v   
        
        return u_coarse, v_coarse
    
class CoarsenWeighted(Operator):
    def __init__(self):
        super().__init__()
    def __call__(self, u, v, ds_hires, ds_coarse):
        '''
        Algorithm: 
        * Interpolate velocities to the center
        * Coarsegrain
        * Interpolate to nodes of Arakawa-C grid on coarse grid

        Note: we weight here all operations with local grid area
        '''

        coarsen = lambda x: x.coarsen({'xh':ds_coarse.factor, 'yh':ds_coarse.factor}).sum()
        
        ############ U-velocity ############
        areaU = ds_hires.param.dxCu * ds_hires.param.dyCu
        u_weighted = ds_hires.grid.interp(u * areaU,'X') * ds_hires.param.wet
        
        areaU = ds_coarse.param.dxCu * ds_coarse.param.dyCu
        u_coarse = ds_coarse.grid.interp(
            coarsen(u_weighted) * ds_coarse.param.wet,'X') \
            * ds_coarse.param.wet_u / areaU

        ############ V-velocity ############
        areaV = ds_hires.param.dxCv * ds_hires.param.dyCv
        v_weighted = ds_hires.grid.interp(v * areaV,'Y') * ds_hires.param.wet
        
        areaV = ds_coarse.param.dxCv * ds_coarse.param.dyCv
        v_coarse = ds_coarse.grid.interp(
            coarsen(v_weighted) * ds_coarse.param.wet,'Y') \
            * ds_coarse.param.wet_v / areaV
                    
        return u_coarse, v_coarse
    
class CoarsenKochkov(Operator):
    def __init__(self):
        super().__init__()
    def __call__(self, u, v, ds_hires, ds_coarse):
        '''
        Algorithm: 
        * Apply weighted coarsegraining along cell side
        * Apply subsampling orthogonally to cell side
        
        Note: This coarsegraining allows to satisfy exactly the incompressibility
        and follows from finite-volume approach, Kochkov2021:
        https://www.pnas.org/doi/abs/10.1073/pnas.2101784118 (see their Supplementary)
        '''
        factor = ds_coarse.factor

        u_coarse = (u * ds_hires.param.dyCu).coarsen({'yh': factor}).sum()[{'xq': slice(factor-1,None,factor)}] \
                * ds_coarse.param.wet_u / ds_coarse.param.dyCu
        
        v_coarse = (v * ds_hires.param.dxCv).coarsen({'xh': factor}).sum()[{'yq': slice(factor-1,None,factor)}] \
                    * ds_coarse.param.wet_v / ds_coarse.param.dxCv

        return u_coarse, v_coarse

class Subsampling(Operator):
    def __init__(self):
        super().__init__()
    def __call__(self, u, v, ds_hires, ds_coarse):
        '''
        Algorithm: 
        * Apply interpolation along cell side
        * Apply subsampling orthogonally to cell side
        
        Note: This subsampling was used in 
        Xie 2020
        https://journals.aps.org/prfluids/abstract/10.1103/PhysRevFluids.5.054606,
        see below Eq. 17
        '''
        factor = ds_coarse.factor

        u_coarse = u.interp(yh = ds_coarse.param.yh)[{'xq': slice(factor-1,None,factor)}] * ds_coarse.param.wet_u

        v_coarse = v.interp(xh = ds_coarse.param.xh)[{'yq': slice(factor-1,None,factor)}] * ds_coarse.param.wet_v

        return u_coarse, v_coarse

class Filtering(Operator):
    def __init__(self, FGR=2, shape=gcm_filters.FilterShape.GAUSSIAN):
        super().__init__()
        self.FGR = FGR
        self.shape = shape

    def __call__(self, u, v, ds_hires, ds_coarse):
        '''
        Algorithm:
        * Initialize GCM-filters with a given FGR,
        informing with local cell area and wet mask
        * Apply filter without coarsegraining or subsampling
        '''
        # Becayse FGR is given w.r.t. coarse grid step
        FGR = self.FGR * ds_coarse.factor

        ############ U-velocity ############
        areaU = ds_hires.param.dxCu * ds_hires.param.dyCu
        filter_simple_fixed_factor = gcm_filters.Filter(
            filter_scale=FGR,
            dx_min=1,
            filter_shape=self.shape,
            grid_type=gcm_filters.GridType.REGULAR_WITH_LAND_AREA_WEIGHTED,
            grid_vars={'area': areaU, 'wet_mask': ds_hires.param.wet_u}
            )
        u_filtered = filter_simple_fixed_factor.apply(u, dims=['yh', 'xq'])

        ############ V-velocity ############
        areaV = ds_hires.param.dxCv * ds_hires.param.dyCv
        filter_simple_fixed_factor = gcm_filters.Filter(
            filter_scale=FGR,
            dx_min=1,
            filter_shape=self.shape,
            grid_type=gcm_filters.GridType.REGULAR_WITH_LAND_AREA_WEIGHTED,
            grid_vars={'area': areaV, 'wet_mask': ds_hires.param.wet_v}
            )
        v_filtered = filter_simple_fixed_factor.apply(v, dims=['yq', 'xh'])

        return u_filtered, v_filtered