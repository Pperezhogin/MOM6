import numpy as np
from helpers.state_functions import Coriolis

def grid_step(self):
    '''
    Here self is the instance of StateFunctions() class
    '''
    dx_center = np.sqrt(self.param.dxT * self.param.dyT)
    dx_corner = np.sqrt(self.param.dxBu * self.param.dyBu)

    # We normalize feature by 50km which is typical grid spacing
    transformation = lambda x: x / 50000
    return transformation(dx_corner), transformation(dx_center)

def deformation_radius(self):
    '''
    Here self is the instance of state_functions class
    '''
    Ld_center = self.data.deformation_radius
    Ld_corner = self.grid.interp(Ld_center * self.param.wet, ['X','Y'])
    
    # We assume that deformation radius varies from 0.01km (1e+2m) to 500km (5e+5m)
    transformation = lambda x: np.log10(np.maximum(1e+1,np.minimum(5e+5,x)))

    return transformation(Ld_corner), transformation(Ld_center)

def deformation_radius_linear(self):
    '''
    Here self is the instance of state_functions class
    '''
    Ld_center = self.data.deformation_radius
    Ld_corner = self.grid.interp(Ld_center * self.param.wet, ['X','Y'])
    
    # We assume that deformation radius varies from 0.01km (1e+2m) to 500km (5e+5m)
    transformation = lambda x: np.minimum(x / (5e+5),1)

    return transformation(Ld_corner), transformation(Ld_center)

def deformation_radius_over_grid_spacing(self):
    '''
    Here self is the instance of state_functions class
    '''
    dx = np.sqrt(self.param.dxT**2 + self.param.dyT**2)
    Ld = self.data.deformation_radius

    Ld_dx_center = Ld / dx
    Ld_dx_corner = self.grid.interp(Ld_dx_center * self.param.wet, ['X','Y'])

    transformation = lambda x: np.log10(np.maximum(1e-3,np.minimum(10,x)))
    
    return transformation(Ld_dx_corner), transformation(Ld_dx_center)

def deformation_radius_over_grid_spacing_linear(self):
    '''
    Here self is the instance of state_functions class
    '''
    dx = np.sqrt(self.param.dxT**2 + self.param.dyT**2)
    Ld = self.data.deformation_radius

    Ld_dx_center = Ld / dx
    Ld_dx_corner = self.grid.interp(Ld_dx_center * self.param.wet, ['X','Y'])

    transformation = lambda x: np.minimum(x / 10, 1)
    
    return transformation(Ld_dx_corner), transformation(Ld_dx_center)

def square_root_of_Ri(self):
    '''
    Here self is the instance of state_functions class
    '''
    f = np.abs(Coriolis(self.param.yh))
    
    sqrtRi_center = f * self.data.eady_time
    sqrtRi_corner = self.grid.interp(sqrtRi_center * self.param.wet, ['X', 'Y'])
    
    transformation = lambda x: np.log10(np.maximum(1e-1,np.minimum(1e+3,x)))
    return transformation(sqrtRi_corner), transformation(sqrtRi_center)

def Held_Larichev_1996(self):
    '''
    Here self is the instance of state_functions class
    '''
    _, beta = Coriolis(self.param.yh, compute_beta = True)
    beta = np.abs(beta)

    feature_center = beta * self.data.deformation_radius * self.data.eady_time
    feature_corner = self.grid.interp(feature_center * self.param.wet, ['X', 'Y'])
    
    transformation = lambda x: np.log10(np.maximum(1e-6,np.minimum(1e+2,x)))
    return transformation(feature_corner), transformation(feature_center)

def Held_Larichev_1996_linear(self):
    '''
    Here self is the instance of state_functions class
    '''
    _, beta = Coriolis(self.param.yh, compute_beta = True)
    beta = np.abs(beta)

    feature_center = beta * self.data.deformation_radius * self.data.eady_time
    feature_corner = self.grid.interp(feature_center * self.param.wet, ['X', 'Y'])
    
    transformation = lambda x: np.minimum(x / (1e+2), 1)

    return transformation(feature_corner), transformation(feature_center)

def Held_Larichev_1996_linear_inverse(self):
    '''
    Here self is the instance of state_functions class
    '''
    _, beta = Coriolis(self.param.yh, compute_beta = True)
    beta = np.abs(beta)

    feature_center = 1 / (beta * self.data.deformation_radius * self.data.eady_time + 1e-10)
    feature_corner = self.grid.interp(feature_center * self.param.wet, ['X', 'Y'])
    
    transformation = lambda x: np.minimum(x / (1e+6), 1)
    
    return transformation(feature_corner), transformation(feature_center)

def Held_Larichev_1996_linear_inverse_range(self):
    '''
    Here self is the instance of state_functions class
    '''
    _, beta = Coriolis(self.param.yh, compute_beta = True)
    beta = np.abs(beta)

    feature_center = 1 / (beta * self.data.deformation_radius * self.data.eady_time + 1e-10)
    feature_corner = self.grid.interp(feature_center * self.param.wet, ['X', 'Y'])
    
    transformation = lambda x: np.minimum(x / (1e+2), 1)
    
    return transformation(feature_corner), transformation(feature_center)

def rescaled_depth(self):
    '''
    Here self is the instance of state_functions class
    '''

    depth_center = self.data.rescaled_depth
    depth_corner = self.grid.interp(depth_center * self.param.wet, ['X', 'Y'])
    
    return depth_corner, depth_center