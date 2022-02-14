import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

class dataset:
    def __init__(self, folder):
        self.series = xr.open_dataset(folder+'/ocean.stats.nc', decode_times=False)
        self.ave = xr.open_mfdataset(folder+'/ave_*.nc', decode_times=False)
        self.prog = xr.open_mfdataset(folder+'/prog_*.nc', decode_times=False)
        self.param = xr.open_dataset(folder+'/ocean_geometry.nc', decode_times=False)
class dataset_experiments:
    def __init__(self, common_folder, exps):
        self.common_folder = common_folder
        self.exps = exps

        self.ds = {}
        for i in range(len(exps)):
            folder = common_folder + '/' + exps[i] + '/output'
            self.ds[exps[i]] = dataset(folder)

    def __getitem__(self, q):
        try:
            return self.ds[q]
        except:
            print('item not found')

    def plot_KE(self, exps, tstart=3650.):
        fig = plt.figure(figsize=(13,5))
        plt.rcParams.update({'font.size': 18})
        for exp in exps:
            plt.subplot(121)
            series = self[exp].series
            t = series.Time
            KE = series.KE.isel(Layer=0) / series.Mass
            KE_mean = KE[t >= tstart].mean()
            p = plt.plot(t/365, KE, label=exp)
            color = p[0].get_color()
            plt.axhline(y = KE_mean, linestyle='--', color=color)
            plt.xlabel('Time, years')
            plt.ylabel('$m^2/s^2$')
            plt.title('Kinetic Energy, upper layer')   
            plt.legend()
            
            plt.subplot(122)
            series = self[exp].series
            t = series.Time
            KE = series.KE.isel(Layer=1) / series.Mass
            KE_mean = KE[t >= tstart].mean()
            p = plt.plot(t/365, KE, label=exp)
            color = p[0].get_color()
            plt.axhline(y = KE_mean, linestyle='--', color=color)
            plt.xlabel('Time, years')
            plt.ylabel('$m^2/s^2$')
            plt.title('Kinetic Energy, lower layer') 
        plt.tight_layout()

    def plot_ssh(self, exps, tstart=3650.):
        nfig = len(exps)
        plt.rcParams.update({'font.size': 16})

        if nfig > 3:
            xfig = int(nfig / 2)
            yfig = 2
        else:
            xfig = nfig
            yfig = 1

        fig = plt.figure(figsize=(xfig*4,yfig*4))

        for ifig, exp in enumerate(exps):
            plt.subplot(int(str(yfig)+str(xfig)+str(ifig+1)))
            ave = self[exp].ave
            t = ave.Time
            ssh = ave.e.isel(zi=0)[t > tstart].mean(dim='Time')
            xh = ave.xh
            yh = ave.yh
            X, Y = np.meshgrid(xh,yh)
            ax = plt.gca()
            Cplot = plt.contour(X,Y,ssh, levels=np.arange(-4,4,0.5), colors='k', linewidths=1)
            ax.clabel(Cplot, Cplot.levels)
            plt.xticks((0, 5, 10, 15, 20))
            plt.yticks((30, 35, 40, 45, 50))
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title(exp)

        plt.tight_layout()

    def plot_relative_vorticity_snapshot(self, exps, Time=-1, zl=0):
        nfig = len(exps)
        plt.rcParams.update({'font.size': 16})

        if nfig > 3:
            xfig = int(nfig / 2)
            yfig = 2
        else:
            xfig = nfig
            yfig = 1

        fig = plt.figure(figsize=(xfig*4,yfig*4))

        for ifig, exp in enumerate(exps):
            plt.subplot(int(str(yfig)+str(xfig)+str(ifig+1)))
            prog = self[exp].prog
            RV = np.array(prog.RV.isel(zl=zl, Time=Time))
            param = self[exp].param
            f = np.array(param.f)
            xq = prog.xq
            yq = prog.yq
            plt.imshow(RV / f, origin='lower', extent=[xq.min(),xq.max(),yq.min(),yq.max()], cmap='inferno')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title(exp)
            plt.clim(-0.2,0.2)

        plt.tight_layout()