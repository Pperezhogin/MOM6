import numpy as np
import math
import xrft
import numpy.fft as npfft
from scipy import signal

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

def select_LatLon(array, Lat, Lon):
    '''
    array is xarray
    Lat, Lon = tuples of floats
    '''
    x = x_coord(array)
    y = y_coord(array)

    return array.sel({x.name: slice(Lon[0],Lon[1]), 
                      y.name: slice(Lat[0],Lat[1])})

def remesh(input, target):
    '''
    Input and target should be xarrays of any type (u-array, v-array, q-array, h-array).
    Datasets are prohibited.
    Horizontal mesh of input changes according to horizontal mesh of target.
    Other dimensions are unchanged!

    If type of arrays is different:
        - Interpolation to correct points occurs
    If input is Hi-res:
        - Coarsening with integer grain and subsequent interpolation to correct mesh if needed
    if input is Lo-res:
        - Interpolation to Hi-res mesh occurs

    Input and output Nan values are treates as zeros (see "fillna")
    '''

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
    
    # B.C.
    result = input.fillna(0)
    
    if (ratiox > 1 or ratioy > 1):
        # Coarsening; x_input.name returns 'xq' or 'xh'
        result = result.coarsen({x_input.name: ratiox, y_input.name: ratioy}, boundary='pad').mean()

    # Coordinate points could change after coarsegraining
    x_result = x_coord(result)
    y_result = y_coord(result)

    # Interpolate if needed
    if not x_result.equals(x_target) or not y_result.equals(y_target):
        result = result.interp({x_result.name: x_target, y_result.name: y_target}).fillna(0)

    # Remove unnecessary coordinates
    if x_target.name != x_input.name:
        result = result.drop_vars(x_input.name)
    if y_target.name != y_input.name:
        result = result.drop_vars(y_input.name)
    
    return result

def compute_2dfft(array, dx, dy, window='hann', Lat=(30,50), Lon=(0,22)):
    '''
    Takes xarray, which must have 2 horizontal coordinates.
    dx, dy are coordinate arrays in metres (Not Lat-Lon). Are used to define mesh over wavenumbers.
    Window: any of https://docs.scipy.org/doc/scipy/reference/signal.windows.html
    Prefer: 'boxcar' - no window, or 'hann'
    Lat-Lon are tuple describing range of coordinates for transform
    Returns analog of rfftn (Real 2d Fourier Transfrom) with coordinates
    (wavenumbers).

    Note some changes to original algorithm (dataset.py) are due to the 
    difference in isotropization function
    '''
    x = x_coord(array)
    y = y_coord(array)

    # select subset of data
    array_ = select_LatLon(array, Lat, Lon).fillna(0) # Remove NaNs
    dx_ = select_LatLon(dx, Lat, Lon).mean().data
    dy_ = select_LatLon(dy, Lat, Lon).mean().data

    nx = len(array_.coords[x.name])
    ny = len(array_.coords[y.name])
    
    # Analog of numpy.fft.rfftn along horizontal coordinates, with subtracting mean
    # and applying window
    arrayf = xrft.fft(array_, dim=(x.name,y.name), shift=False, window=window, detrend='constant')    
    
    # select only positive frequencies in k_x wavenumber
    x_freq = arrayf.coords['freq_'+x.name] # get x frequency coordinate
    y_freq = arrayf.coords['freq_'+y.name] # get y frequency coordinate
    #arrayf = arrayf.sel({x_freq.name: slice(0,x_freq.max())}) # original method, which removes half of frequencies
    

    # Define wavenumbers according to mean mesh step
    arrayf = arrayf.rename({x_freq.name: 'kx', y_freq.name: 'ky'})
    #kx = npfft.rfftfreq(nx,d=dx_/(2*np.pi)) # if half of requencies are removed
    kx = npfft.fftfreq(nx,d=dx_/(2*np.pi))
    ky = npfft.fftfreq(ny,d=dy_/(2*np.pi))
    arrayf['kx'] = kx
    arrayf['ky'] = ky

    # renormalize according to the window size and number of mesh points
    Wx = signal.windows.__getattribute__(window)(nx)
    Wy = signal.windows.__getattribute__(window)(ny)
    Wnd = np.outer(Wx, Wy)
    Wnd_sqr = (Wnd**2).mean()
    arrayf = arrayf / (nx*ny) / np.sqrt(Wnd_sqr) # original normalization as in dataset.py
    arrayf /= np.sqrt(np.diff(kx)[0] * np.diff(ky)[0]) # see 25 Jan comment https://github.com/pyqg/pyqg/issues/275
    return arrayf