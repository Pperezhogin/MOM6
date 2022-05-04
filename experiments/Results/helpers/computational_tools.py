import numpy as np
import math

def remesh(input, target):
    '''
    Input and target should be xarrays of any type (u-array, v-array, q-array, h-array)
    The result must have the same mesh as target, but the data should correspond to input

    If type of arrays is different:
        - Interpolation to correct points occurs
    If input is Hi-res:
        - Coarsening with integer grain and subsequent interpolation to correct mesh if needed
    if input is Lo-res:
        - Interpolation to Hi-res mesh occurs

    Input and output Nan values are treates as zeros (see "fillna")
    '''

    def x_coord(array):
        try:
            coord = array.xq
        except:
            coord = array.xh
        return coord
    
    def y_coord(array):
        try:
            coord = array.yq
        except:
            coord = array.yh
        return coord

    # Define coordinates
    x_input  = x_coord(input)
    y_input  = y_coord(input)
    x_target = x_coord(target)
    y_target = y_coord(target)

    # ratio of mesh steps
    ratiox = np.diff(x_target)[0] / np.diff(x_input)[0]
    ratiox = math.ceil(ratiox)

    ratioy = np.diff(y_target)[0] / np.diff(y_input)[0]
    ratioy = math.ceil(ratioy)

    # Coarsening; x_input.name returns 'xq' or 'xh'
    result = input.fillna(0).coarsen({x_input.name: ratiox, y_input.name: ratioy}, boundary='pad').mean()

    # Interpolating to target mesh
    result = result.interp({x_input.name: x_target, y_input.name: y_target}).fillna(0)

    # Remove unnecessary coordinates
    target_set = set(target.coords)
    result_set = set(result.coords)

    return result.drop_vars(result_set-target_set)