import os
import xarray as xr

class netcdf_property:
    '''
    Implements storage of statistical characteristic
    in experiment folder at netcdf file
    having name KE_key.nc,
    where key - additional usually name of the experiment
    Information about folder in decorated class. See:
        instance.folder
        instance.key
    '''
    def __init__(self, function):
        self.function = function
    
    def __get__(self, instance, owner):
        '''
        Method __get__ is called, when this class(netcdf_cache)
        is accessed as attribute of abother once class (owner),
        or its instance (instance)
        https://python-reference.readthedocs.io/en/latest/docs/dunderdsc/get.html
        '''        
        if instance is None: return self # see https://gist.github.com/asross/952fa456f8bcd07abf684cc515d49030

        funcname = self.function.__name__
        #filename = os.path.join(instance.folder, funcname+'_'+instance.key+'.nc')
        #filename = os.path.join('/scratch/pp2681/mom6/cache', funcname+'_'+instance.key+'.nc')
        filename = os.path.join('/scratch/pp2681/mom6/cache', '-'.join(instance.folder.split('/')[4:-2])+'-'+instance.key+'-'+funcname+'.nc')
        #print(filename)
        if instance.recompute:
            try:
                os.remove(filename)
                #print(f'Removing cache file {filename}')
            except:
                pass

        # Try to open netcdf if exists
        if os.path.exists(filename):
            #print(f'Reading file {filename}')
            try:
                ncfile = xr.open_dataset(filename, decode_times=False, chunks={'Time': 1, 'zl': 1})
            except:
                ncfile = xr.open_dataset(filename, decode_times=False) # for very small files
            #print(f'Returning cached value of {funcname}')
            value = ncfile[funcname]
            ncfile.close() # to prevent out of memory
            return value

        #print(f'Calculating value of {funcname}')
        try:
            value = self.function(instance).chunk({'Time':1,'zl':1})
        except:
            value = self.function(instance) # for very small objects
        
        # Create new dataset
        ncfile = xr.Dataset()
        
        # store on disk and close file
        ncfile[funcname] = value
        #print(f'Saving result to {filename}')
        ncfile.to_netcdf(filename)
        ncfile.close() # to prevent out of memory

        return value