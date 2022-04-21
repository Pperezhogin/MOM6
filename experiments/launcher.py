import os
import math
import json

# creates slurm script mom.sub
def create_slurm(p, filename):
    # p - dictionary with parameters
    lines = [
    '#!/bin/bash',
    '#SBATCH --nodes='+str(p['nodes']),
    '#SBATCH --ntasks-per-node='+str(p['ntasks']),
    '#SBATCH --cpus-per-task=1',
    '#SBATCH --mem='+str(p['mem'])+'GB',
    '#SBATCH --time='+str(p['time'])+':00:00',
    '#SBATCH --job-name='+str(p['name']),
    'module purge',
    'source ~/MOM6-examples/build/intel/env',
    'mpiexec ./MOM6 > out.txt',
    'mkdir -p output',
    'mv *.nc output'
    ]
    with open(filename,'w') as fid:
        fid.writelines([ line+'\n' for line in lines])

# configuration of resolutions
def R_lines(resolution):
    if resolution == 'R2':
        return ['#override NIGLOBAL = 44',
            '#override NJGLOBAL = 40',
            '#override DT = 2160.',
            '#override DT_FORCING = 2160.'
        ]
    if resolution == 'R3':
        return ['#override NIGLOBAL = 66',
            '#override NJGLOBAL = 60',
            '#override DT = 1440.',
            '#override DT_FORCING = 1440.'
        ]
    if resolution == 'R4':
        return [
            '#override NIGLOBAL = 88',
            '#override NJGLOBAL = 80',
            '#override DT = 1080.',
            '#override DT_FORCING = 1080.'
        ]
    if resolution == 'R6':
        return [
            '#override NIGLOBAL = 132',
            '#override NJGLOBAL = 120',
            '#override DT = 720.',
            '#override DT_FORCING = 720.'
        ]
    if resolution == 'R8':
        return [ '#override NIGLOBAL = 176',
            '#override NJGLOBAL = 160',
            '#override DT = 540.',
            '#override DT_FORCING = 540.'
        ]
    if resolution == 'R10':
        return [
            '#override NIGLOBAL = 220',
            '#override NJGLOBAL = 200',
            '#override DT = 432.',
            '#override DT_FORCING = 432.'
        ]
    if resolution == 'R12':
        return [
            '#override NIGLOBAL = 264',
            '#override NJGLOBAL = 240',
            '#override DT = 360.',
            '#override DT_FORCING = 360.'
        ]
    if resolution == 'R16':
        return [
            '#override NIGLOBAL = 352',
            '#override NJGLOBAL = 320',
            '#override DT = 270.',
            '#override DT_FORCING = 270.'
        ]
    if resolution == 'R32':
        return [
            '#override NIGLOBAL = 704',
            '#override NJGLOBAL = 640',
            '#override DT = 135.',
            '#override DT_FORCING = 135.'
        ]
    if resolution == 'R64':
        return [
            '#override NIGLOBAL = 1408',
            '#override NJGLOBAL = 1280',
            '#override DT = 67.5',
            '#override DT_FORCING = 67.5'
        ]

def create_MOM_override(p, filename):
    # p - dictionary of parameters
    lines = []
    for key in p.keys():
        if key == 'resolution':
            lines.extend(R_lines(p[key]))
        else:
            lines.append('#override '+key+' = '+str(p[key]))
    with open(filename,'w') as fid:
        fid.writelines([ line+'\n' for line in lines])

def queue_experiment(folder, hpc, parameters):
    os.system('rm -rf '+folder)
    os.system('mkdir -p '+folder)
    
    create_slurm(hpc, os.path.join(folder,'mom.sub'))
    create_MOM_override(parameters, os.path.join(folder,'MOM_override'))
    
    os.system('cp -r ~/MOM6-examples/src/MOM6/experiments/configurations/double_gyre/* '+folder)
    os.system('cp ~/MOM6-examples/build/intel/ocean_only/repro/MOM6 '+folder)

    with open(folder+'/args.json', 'w') as f:
        json.dump(parameters, f, indent=2)
    
    os.system('cd '+folder+'; sbatch mom.sub')

#############################################################################################

hpc = {
    'nodes': 1,
    'ntasks': 16,
    'mem': 16,
    'time': 1,
    'name': 'EXP1'
}

parameters = {'resolution': 'R2',
     'DAYMAX': 7300.0,
     'RESTINT': 1825.0,
     'LAPLACIAN': 'False',
     'BIHARMONIC': 'True',
     'SMAGORINSKY_AH': True,
     'SMAG_BI_CONST': 0.06, 
     'USE_ZB2020': 'True', 
     'amplitude': 1., 
     'ZB_type': 0, 
     'ZB_cons': 1, 
     'LPF_iter': 0, 
     'LPF_order': 1, 
     'HPF_iter': 0,
     'HPF_order': 0,
     'Stress_iter': 4,
     'Stress_order': 4}

queue_experiment('/scratch/pp2681/mom6/test',hpc,parameters)