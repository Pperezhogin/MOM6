import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as functional
import numpy as np
import os
import xarray as xr
from time import time

def image_to_3x3_stencil(x):
    '''
    Note that x is the fast direction in CM2.6 and 
    also it is the rightmost.
    Transforms image data to Npoints * 9 data. 
    The input and output are torch tensors
    '''
    ny, nx = x.shape
    # Single feature has dimension Npoints x 9,
    # where 9 corresponds to stencil 3x3, and 
    # zonal direction is the fast one
    y = torch.zeros((nx-2)*(ny-2),9)
    k = 0
    for j in range(1,x.shape[0]-1):
        for i in range(1,x.shape[1]-1):
            y[k,:] = x[j-1:j+2,i-1:i+2].reshape(1,9)
            k += 1
    return y

def image_to_3x3_stencil_gpt(x):
    '''
    It is the same function, but 1000 times faster
    suggested by gpt. The result is the same up to 
    float32 precision (i.e., it is not exact)
    '''
    ny, nx = x.shape
    patches = x.unfold(0, 3, 1).unfold(1, 3, 1).reshape(-1, 9)
    return patches

def log_to_xarray(log_dict):
    anykey = list(log_dict.keys())[0]
    num_epochs = len(log_dict[anykey])
    epoch = coord(np.arange(1, num_epochs+1), 'epoch')
    for key in log_dict.keys():
        log_dict[key] = xr.DataArray(log_dict[key], dims='epoch', coords=[epoch])
        
    return xr.Dataset(log_dict)

class ANN(nn.Module):
    def __init__(self, layer_sizes=[3, 17, 27, 5], output_norm=None):
        super().__init__()
        
        self.layer_sizes = layer_sizes
        self.output_norm = output_norm  

        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i < len(self.layers)-1:
                x = functional.relu(x)
        return x
    
    def compute_loss(self, x, ytrue):
        MSE = nn.MSELoss()(self.forward(x), ytrue)
        if self.output_norm is not None:
            MSE = MSE / self.output_norm**2
        return {'loss': MSE}
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

def export_ANN(ann, input_norms, output_norms, filename='ANN_test.nc'):
    ds = xr.Dataset()
    ds['num_layers'] = xr.DataArray(len(ann.layer_sizes)).expand_dims('dummy_dimension')
    ds['layer_sizes'] = xr.DataArray(ann.layer_sizes, dims=['nlayers'])
    ds = ds.astype('int32') # MOM6 reads only int32 numbers
    
    for i in range(len(ann.layers)):
        # Naming convention for weights and dimensions
        matrix = f'A{i}'
        bias = f'b{i}'
        ncol = f'ncol{i}'
        nrow = f'nrow{i}'
        layer = ann.layers[i]
        
        # Transposed, because torch is row-major, while Fortran is column-major
        ds[matrix] = xr.DataArray(layer.weight.data.T, dims=[ncol, nrow])
        ds[bias] = xr.DataArray(layer.bias.data, dims=[nrow])
    
    # Save true answer for random vector for testing
    x0 = torch.randn(ann.layer_sizes[0])
    y0 = ann(x0 / input_norms) * output_norms
    nrow = f'nrow{len(ann.layers)-1}'
    
    ds['x_test'] = xr.DataArray(x0.data, dims=['ncol0'])
    ds['y_test'] = xr.DataArray(y0.data, dims=[nrow])
    
    ds['input_norms']  = xr.DataArray(input_norms.data, dims=['ncol0'])
    ds['output_norms'] = xr.DataArray(output_norms.data, dims=[nrow])

    
    print('x_test = ', ds['x_test'].data)
    print('y_test = ', ds['y_test'].data)
    
    if os.path.exists(filename):
        print(f'Rewrite {filename} ?')
        input()
        os.system(f'rm -f {filename}')
        print(f'{filename} is rewritten')
    
    ds.to_netcdf(filename)
    
def import_ANN(filename='ANN_test.nc'):
    ds = xr.open_dataset(filename)
    layer_sizes = ds['layer_sizes'].values

    ann = ANN(layer_sizes)
    
    for i in range(len(ann.layers)):
        # Naming convention for weights and dimensions
        matrix = torch.tensor(ds[f'A{i}'].T.values) # Transpose back: it is convention
        bias = torch.tensor(ds[f'b{i}'].values)
        ann.layers[i].weight.data = matrix
        ann.layers[i].bias.data = bias
        
    x_test = torch.tensor(ds['x_test'].values.reshape(1,-1))
    y_pred = ann(x_test).detach()
    y_test = ds['y_test'].values
    
    rel_error = float(np.abs(y_pred - y_test).max() / np.abs(y_test).max())
    if rel_error > 1e-6:
        print(f'Test prediction using {filename}: {rel_error}')
    return ann

class AverageLoss():
    '''
    Accumulates dictionary of losses over batches
    and computes mean for epoch.
    List of keys to accumulate given by 'losses'
    Usage:
    Init before epoch. 
    Accumulate over batches for given epoch
    Average after epoch
    '''
    def __init__(self, log_dict):
        self.init_me = True
        self.count = {}

    def accumulate(self, log_dict, losses, n: int):
        '''
        log_dict: dictionary of timeseries
        losses: dictionary of loss on a batch
        n: number of elements in batch
        '''
        keys = losses.keys()
        if (self.init_me):
            new_keys = set(losses.keys())-set(log_dict.keys())
            for key in new_keys:
                log_dict[key] = []
            for key in keys:
                self.count[key] = 0
                log_dict[key].append(0.)
            self.init_me = False

        for key in keys:
            value = losses[key]
            # extract floats from scalar tensors
            if isinstance(value, torch.Tensor):
                try:
                    value = value.item()
                except:
                    value = value.cpu().numpy()
            log_dict[key][-1] += value * n
            self.count[key] += n
    
    def average(self, log_dict):
        '''
        Updates last element of dictionary with 
        average value
        '''
        for key in self.count.keys():
            log_dict[key][-1] = log_dict[key][-1] / self.count[key]

def dict_postfix(mydict, postfix):
    return {str(key)+postfix: val for key, val in mydict.items()}

def minibatch(*arrays: np.array, batch_size=64, shuffle=True):
    '''
    Recieves arbitrary number of numpy arrays
    of size 
    Nbatch x Nfeatures x Ny x Nx.
    Returns multiple batches of tensors of size 
    batch_size x Nfeatures x Ny x Nx.
    '''
    assert len(set([len(a) for a in arrays])) == 1
    order = np.arange(len(arrays[0]))
    if shuffle:
        np.random.shuffle(order)
    steps = int(np.ceil(len(arrays[0]) / batch_size))
    for step in range(steps):
        idx = order[step*batch_size:(step+1)*batch_size]
        yield tuple(torch.as_tensor(array[idx]) for array in arrays)

def evaluate_test(net, *arrays: np.array, batch_size=64, postfix='_test', device=None):
    '''
    Updates logger on test dataset
    '''
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # To do not update parameters of 
    # Batchnorm2d layers
    net.eval()

    logger = AverageLoss(net.log_dict)
    for xy in minibatch(*arrays, batch_size=batch_size):
        with torch.no_grad():
            losses = net.compute_loss(*[x.to(device) for x in xy])
        logger.accumulate(net.log_dict, dict_postfix(losses, postfix), len(xy[0]))
    
    logger.average(net.log_dict)
    net.train()

def train(net, X_train: np.array, Y_train: np.array, 
        X_test: np.array, Y_test: np.array, 
        num_epochs, batch_size, learning_rate, device=None,
        normalize_loss=False, print_frequency=1):
    '''
    X_train, Y_train are arrays of size
    Nbatch x Nfeatures x Ny x Nx.
    For this function to use, class 'net'
    should implement function compute_loss(x,y) returning 
    dictionary, where key 'loss'
    is used for optimization,
    while others are used for logger.
    '''
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if device != "cpu":
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    else:
        device_name = "cpu"
    
    if print_frequency > 0:
        print(f"Training starts on device {device_name}, number of samples {len(X_train)}")
    
    net.to(device)
    # Switch batchnorm2d layer to training mode
    net.train()

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
        milestones=[int(num_epochs/2), int(num_epochs*3/4), int(num_epochs*7/8)], gamma=0.1)  

    try:
        net.log_dict
    except:
        net.log_dict = {}
            
    t_s = time()
    for epoch in range(0,num_epochs):
        t_e = time()
        logger = AverageLoss(net.log_dict)
        for x, y in minibatch(X_train, Y_train, batch_size=batch_size):
        # for x, y in data_loader:
            optimizer.zero_grad()
            losses = net.compute_loss(x.to(device),y.to(device))
            if normalize_loss:
                losses['loss'] = losses['loss'] / losses['loss'].detach()
            losses['loss'].backward() # optimize over the 'loss' value
            optimizer.step()
            logger.accumulate(net.log_dict, losses, len(x))
        scheduler.step()

        logger.average(net.log_dict)
        evaluate_test(net, X_test, Y_test, batch_size=batch_size, device=device)
        t = time()
        if epoch % print_frequency == 0 and print_frequency > 0:
            print('[%d/%d] [%.2f/%.2f] Loss: [%.6f, %.6f]' 
                % (epoch+1, num_epochs,
                t-t_e, (t-t_s)*(num_epochs/(epoch+1)-1),
                net.log_dict['loss'][-1], net.log_dict['loss_test'][-1]))