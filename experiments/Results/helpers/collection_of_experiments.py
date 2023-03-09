import xarray as xr
import os
from helpers.experiment import Experiment
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

class CollectionOfExperiments:
    '''
    This class extend dictionary of experiments by additional
    tools for plotting and comparing experiments
    '''
    def __init__(self, exps, experiments_dict, names_dict):
        '''
        experiments_dict - "experiment" objects labeled by keys
        names_dict - labels for plotting
        '''
        self.exps = exps
        self.experiments = experiments_dict
        self.names = names_dict

    def __getitem__(self, q):
        ''' 
        Access experiments with key values directly
        '''
        try:
            return self.experiments[q]
        except:
            print('item not found')
    
    def __add__(self, otherCollection):
        # merge dictionaries and lists
        exps = [*self.exps, *otherCollection.exps]
        experiments_dict = {**self.experiments, **otherCollection.experiments}
        names_dict = {**self.names, **otherCollection.names}

        return CollectionOfExperiments(exps, experiments_dict, names_dict)

    def compute_statistics(self, exps=None, recompute=False):
        if exps is None:
            exps = self.exps
        for exp in exps:
            if recompute:
                self[exp].recompute = True
            for key in Experiment.get_list_of_netcdf_properties():
                self[exp].__getattribute__(key)
            self[exp].recompute = False

    def remesh(self, input, target, exp=None, name=None, compute=False):
        '''
        input  - key of experiment to coarsegrain
        target - key of experiment we want to take coordinates from
        '''

        if exp is None:
            exp = input+'_'+target
        if name is None:
            name = input+' coarsegrained to '+target

        result = self[input].remesh(self[target], exp, compute) # call experiment method

        print('Experiment '+input+' coarsegrained to '+target+
            ' is created. Its identificator='+exp)
        self.exps.append(exp)
        self.experiments[exp] = result
        self.names[exp] = name
    
    @classmethod
    def init_folder(cls, common_folder, exps=None, exps_names=None, additional_subfolder='', prefix=None):
        '''
        Scan folders in common_folder and returns class instance with exps given by these folders
        exps - list of folders can be specified
        exps_names - list of labels can be specified
        additional_subfolder - if results are stored not in common_folder+exps[i],
        but in an additional subfolder 
        '''

        if exps is None:
            folders = sorted(os.listdir(common_folder))

        if exps_names is None:
            exps_names = folders

        if prefix:
            exps = [prefix+'-'+exp for exp in folders]
        else:
            exps = folders

        # Construct dictionary of experiments, where keys are given by exps
        experiments_dict = {}
        names_dict = {}
        for i in range(len(exps)):
            folder = os.path.join(common_folder,folders[i],additional_subfolder)
            experiments_dict[exps[i]] = Experiment(folder, exps[i])
            names_dict[exps[i]] = exps_names[i] # convert array to dictionary

        return cls(exps, experiments_dict, names_dict)
    
    def plot_KE_spectrum(self, exps, key='KE_spectrum'):
        fig, ax = plt.subplots(1,2,figsize=(13,7))
        p = []
        for exp in exps:
            KE = self[exp].__getattribute__(key)
            k = KE.freq_r

            KE_upper = KE.isel(zl=0)
            KE_lower = KE.isel(zl=1)

            p.extend(ax[0].loglog(k, KE_upper, label=self.names[exp]))
            ax[0].set_xlabel(r'wavenumber, $k [m^{-1}]$')
            ax[0].set_ylabel(r'Energy spectrum, $E(k) [m^3/s^2]$')
            ax[0].set_title('Upper layer')
            ax[0].legend(prop={'size': 14})
            ax[0].grid(which='both',linestyle=':')

            p.extend(ax[1].loglog(k, KE_lower, label=self.names[exp]))
            ax[1].set_xlabel(r'wavenumber, $k [m^{-1}]$')
            ax[1].set_ylabel(r'Energy spectrum, $E(k) [m^3/s^2]$')
            ax[1].set_title('Lower layer')
            ax[1].legend(prop={'size': 14})
            ax[1].grid(which='both',linestyle=':')

        k = [5e-5, 1e-3]
        E = [1.5e+2, 0]
        E[1] = E[0] * (k[1]/k[0])**(-3)
        ax[0].loglog(k,E,'--k')
        ax[0].text(2e-4,1e+1,'$k^{-3}$')
        ax[0].set_xlim([2e-6, 2e-3])
        
        ax[1].set_xlim([2e-6, 2e-3])
        k = [5e-5, 1e-3]
        E = [3e+1, 0]
        E[1] = E[0] * (k[1]/k[0])**(-3)
        ax[1].loglog(k,E,'--k')
        ax[1].text(2e-4,1e+1,'$k^{-3}$')

        return p
    
    def plot_transfer(self, exp, target='R64_R4', callback=True):
        smag = self[exp].Smagorinsky_transfer
        ZB = self[exp].ZB_transfer
        kmax = self[exp].kmax
        if target is not None:
            SGS = self[target].SGS_transfer

        matplotlib.rcParams.update({'font.family': 'MathJax_Main',
        'mathtext.fontset': 'cm','axes.formatter.limits': (-1,2), 
        'axes.formatter.use_mathtext': True, 'font.size': 16})
        plt.figure(figsize=(15,4))
        for zl in range(2):
            plt.subplot(1,2,zl+1)
            if target is not None:
                SGS.isel(zl=zl).plot(label='SGS', color='k', ls='-')
            ZB.isel(zl=zl).plot(label='ZB', color='tab:orange', ls='--')
            smag.isel(zl=zl).plot(label='Smag', color='tab:green', ls='-.')
            (ZB+smag).isel(zl=zl).plot(label='ZB+Smag', color='tab:blue')
            plt.legend()
            plt.axhline(y=0,ls='-',color='gray',alpha=0.5)
            ax2 = plt.gca().secondary_xaxis('top', functions=(lambda x: x/kmax, lambda x: x*kmax))
            ax2.set_xlabel('Frequency/Nyquist')
            ax2.set_ticks([0.25, 0.5, 1],[r'$1/4$', r'$1/2$', r'$1$'])
            for k in [0.25, 0.5, 1]:
                plt.axvline(x=kmax*k,ls='-',color='gray',alpha=0.5)
            plt.xlim([None, kmax])
            plt.xlabel('wavenumber $k$ [m$^{-1}$]')
            plt.ylabel('KE transfer [m$^3$/s$^3$]')
            if zl==0:
                plt.title('Upper layer',fontweight='bold',fontsize=25, loc='right')
                plt.title('')
            else:
                plt.title('Lower layer',fontweight='bold',fontsize=25, loc='right')
                plt.title('')

        if callback:
            self.plot_power(exp,target)

    def plot_power(self, exp, target='R64_R4'):
        smag = self[exp].Smagorinsky_power
        ZB = self[exp].ZB_power
        model = self[exp].Model_power
        kmax = self[exp].kmax
        if target is not None:
            SGS = self[target].SGS_power

        matplotlib.rcParams.update({'font.family': 'MathJax_Main',
        'mathtext.fontset': 'cm','axes.formatter.limits': (-1,2), 
        'axes.formatter.use_mathtext': True, 'font.size': 16})
        plt.figure(figsize=(15,4))
        for zl in range(2):
            plt.subplot(1,2,zl+1)
            if target is not None:
                SGS.isel(zl=zl).plot(label='SGS', color='k', ls='-')
            ZB.isel(zl=zl).plot(label='ZB', color='tab:orange', ls='--')
            smag.isel(zl=zl).plot(label='Smag', color='tab:green', ls='-.')
            model.isel(zl=zl).plot(label='ZB+Smag', color='tab:blue')
            plt.legend()
            plt.axhline(y=0,ls='-',color='gray',alpha=0.5)
            for k in [0.25, 0.5, 1]:
                plt.axvline(x=kmax*k,ls='-',color='gray',alpha=0.5)
            plt.xlim([None, kmax])
            plt.xlabel('wavenumber $k$ [m$^{-1}$]')
            plt.ylabel('Power spectrum [m$^3$/s$^4$]')
            plt.title('')
            
    def plot_ssh(self, exps):
        plt.figure(figsize=(15,4))
        nfig = len(exps)
        for ifig, exp in enumerate(exps):
            plt.subplot(1,nfig,ifig+1)
            Cplot = self[exp].ssh_mean.plot.contour(levels=np.arange(-4,4,0.5), colors='k', linewidths=1)
            plt.gca().clabel(Cplot, Cplot.levels)
            plt.xticks((0, 5, 10, 15, 20))
            plt.yticks((30, 35, 40, 45, 50))
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title(self.names[exp])

        plt.tight_layout()

    def plot_KE_PE(self, exps=['R4', 'R8', 'R64_R4'], labels=None, color=['k', 'tab:cyan', 'tab:blue', 'tab:red']):
        if labels is None:
            labels = exps
        plt.figure(figsize=(9,9))
        plt.subplots_adjust(wspace=0.4, hspace=0.6)
        width = (len(exps)-1) * [0.4] + [1]
        for zl in range(2):
            plt.subplot(2,2,zl+1)
            MKE = []
            EKE = []
            for exp in exps:          
                MKE.append(1e-15*self[exp].MKE_joul.isel(zl=zl).values)
                EKE.append(1e-15*self[exp].EKE_joul.isel(zl=zl).values)
            x=np.arange(len(exps));
            x[-1] += 1.5
            plt.bar(x,MKE,width,label='MKE',color=color[0])
            plt.bar(x,EKE,width,bottom=MKE,label='EKE',color=color[1])
            plt.ylabel('Kinetic energy, PJ', fontsize=14);
            plt.xticks(ticks=x,labels=labels);
            if zl==0:
                plt.title('KE, Upper layer')
            else:
                plt.title('KE, Lower layer')
            plt.legend(loc='upper left', fontsize=14)
            plt.ylim([0, (EKE[-1]+MKE[-1])*(1.55-zl/2)])
            
        plt.subplot(2,2,3)
        MPE = []
        EPE = []
        for exp in exps:
            MPE.append(1e-15*self[exp].MPE_joul.values)
            EPE.append(1e-15*self[exp].EPE_joul.values)     
        x=np.arange(len(exps));
        x[-1] += 1.5
        plt.bar(x,MPE,width,label='MPE',color=color[2])
        plt.bar(x,EPE,width,bottom=MPE,label='EPE',color=color[3])
        plt.ylabel('Interface displacement \n potential energy, PJ', fontsize=14);
        plt.xticks(ticks=x,labels=labels);
        plt.title('Potential energy')
        plt.legend(loc='upper left', fontsize=14)
        plt.ylim([0, (EPE[-1]+MPE[-1])*1.8])
        plt.axhline(y=MPE[-1], ls=':', color=color[2])
        
        plt.subplot(2,2,4)
        EKE = []
        for exp in exps:          
            EKE.append(1e-15*self[exp].EKE_joul.values.sum())
        x=np.arange(len(exps));
        x[-1] += 1.5
        plt.bar(x,EKE,width,label='EKE',color=color[1])
        plt.bar(x,EPE,width,bottom=EKE, label='EPE',color=color[3])
        plt.ylabel('Eddy energy, PJ', fontsize=14)
        plt.title('Energy of eddies')
        plt.xticks(ticks=x,labels=labels);
        plt.legend(loc='upper left', fontsize=14)
        plt.ylim([0, (EKE[-1]+EPE[-1])*1.4])
