import os
import json
import numpy as np

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

def run_individual_experiment(folder, hpc, parameters):
    os.system('rm -rf '+folder)
    os.system('mkdir -p '+folder)
    
    create_slurm(hpc, os.path.join(folder,'mom.sub'))
    create_MOM_override(parameters, os.path.join(folder,'MOM_override'))
    
    os.system('cp -r ~/MOM6-examples/src/MOM6/experiments/configurations/double_gyre/* '+folder)
    os.system('cp ~/MOM6-examples/build/intel/ocean_only/repro/MOM6 '+folder)

    with open(os.path.join(folder,'args.json'), 'w') as f:
        json.dump(parameters, f, indent=2)
    
    os.system('cd '+folder+'; sbatch mom.sub')

def get_nth_key(dictionary, n=0):
    if n < 0:
        n += len(dictionary)
    for i, key in enumerate(dictionary.keys()):
        if i == n:
            return key
    raise IndexError("dictionary index out of range") 

# converts dictionary of lists to list of dictionaries
def iterate_dictionary(x, y={}, key_id=0):
    if key_id == 0:
        y = {}
    if key_id == len(x):
        yield y.copy()
        return
    
    key = get_nth_key(x,key_id)
    value = x[key]
    if isinstance(value,list):
        for val in value:
            y[key] = val
            yield from iterate_dictionary(x,y,key_id+1)
    else:
        y[key] = value
        yield from iterate_dictionary(x,y,key_id+1)

# run experiment in common folder. If the number of experiment exceed max_runs, 
# parameters are fetched stochastically
def run_many_experiments(folder, hpc, parameters, EXP_start=None, max_runs=100):
    list_parameters = list(iterate_dictionary(parameters))
    
    nruns = len(list_parameters)
    if (nruns > max_runs):
        print(f'Number of runs ({nruns}) exceeds maximum ({max_runs}).')
        print('Parameters are fetched stochastically.')
        idx = np.arange(nruns)
        np.random.shuffle(idx)
        idx = idx[:max_runs]
        short_list = [list_parameters[i] for i in idx]
    else:
        print('Parameters are fetched deterministically.')
        short_list = list_parameters
    
    os.system('mkdir -p '+folder)
    folders_list = os.listdir(folder)
    folders_list.sort(key=lambda a: int(a[3:]))
    print('')
    print('Already contained experiments in the folder:')
    print(*folders_list)
    print('')

    if EXP_start is None:
        EXP_start = len(folders_list)+1

    print('The number of experiments to schedule:', len(short_list))
    print('Starting folder', 'EXP'+str(EXP_start))
    input('Continue? Press Enter...')

    for i, parameter in enumerate(short_list):
        EXP_name = 'EXP'+str(i+EXP_start)
        hpc['name'] = EXP_name
        run_individual_experiment(os.path.join(folder,EXP_name),hpc,parameter)

    print('All experiments are queued.')
        
#############################################################################################

hpc = {
    'nodes': 1,
    'ntasks': 10,
    'mem': 16,
    'time': 24,
}

# Eddy viscosity models
parameters = {'resolution': 'R4',
     'DAYMAX': 7300.0,
     'RESTINT': 1825.0,
     'LAPLACIAN': 'False',
     'BIHARMONIC': 'True',
     'SMAGORINSKY_AH': True,
     'SMAG_BI_CONST': [i/100 for i in range(1,11)], 
     'USE_ZB2020': 'False',
     'amplitude': 1., 
     'ZB_type': 0, 
     'ZB_cons': 1, 
     'LPF_iter': 0, 
     'LPF_order': 1, 
     'HPF_iter': 0,
     'HPF_order': 1,
     'Stress_iter': 0,
     'Stress_order': 1
}

run_many_experiments('/scratch/pp2681/mom6/Apr2022/R4/', hpc, parameters)

# Bare ZB2020 models
parameters = {'resolution': 'R4',
     'DAYMAX': 7300.0,
     'RESTINT': 1825.0,
     'LAPLACIAN': 'False',
     'BIHARMONIC': 'True',
     'SMAGORINSKY_AH': True,
     'SMAG_BI_CONST': [i/100 for i in range(1,11)], 
     'USE_ZB2020': 'True',
     'amplitude': [i/24 for i in range(1,11)], 
     'ZB_type': 0, 
     'ZB_cons': 1, 
     'LPF_iter': 0, 
     'LPF_order': 1, 
     'HPF_iter': 0,
     'HPF_order': 1,
     'Stress_iter': 0,
     'Stress_order': 1
}

run_many_experiments('/scratch/pp2681/mom6/Apr2022/R4/', hpc, parameters)

# ADM models
parameters = {'resolution': 'R4',
     'DAYMAX': 7300.0,
     'RESTINT': 1825.0,
     'LAPLACIAN': 'False',
     'BIHARMONIC': 'True',
     'SMAGORINSKY_AH': True,
     'SMAG_BI_CONST': [i/100 for i in range(1,11)], 
     'USE_ZB2020': 'True',
     'amplitude': [i/24 for i in range(1,25)], 
     'ZB_type': 0, 
     'ZB_cons': 1, 
     'LPF_iter': [i for i in range(1,5)], 
     'LPF_order': [i for i in range(1,5)], 
     'HPF_iter': 0,
     'HPF_order': 1,
     'Stress_iter': 0,
     'Stress_order': 1
}

run_many_experiments('/scratch/pp2681/mom6/Apr2022/R4/', hpc, parameters, max_runs=20)

# ADM filtered models
parameters = {'resolution': 'R4',
     'DAYMAX': 7300.0,
     'RESTINT': 1825.0,
     'LAPLACIAN': 'False',
     'BIHARMONIC': 'True',
     'SMAGORINSKY_AH': True,
     'SMAG_BI_CONST': [i/100 for i in range(1,11)], 
     'USE_ZB2020': 'True',
     'amplitude': [i/24 for i in range(1,25)], 
     'ZB_type': 0, 
     'ZB_cons': 1, 
     'LPF_iter': [i for i in range(1,5)], 
     'LPF_order': [i for i in range(1,5)], 
     'HPF_iter': 0,
     'HPF_order': 1,
     'Stress_iter': [i for i in range(1,5)],
     'Stress_order': [i for i in range(1,5)]
}

run_many_experiments('/scratch/pp2681/mom6/Apr2022/R4/', hpc, parameters, max_runs=20)

# Perezhogin2019 models
parameters = {'resolution': 'R4',
     'DAYMAX': 7300.0,
     'RESTINT': 1825.0,
     'LAPLACIAN': 'False',
     'BIHARMONIC': 'True',
     'SMAGORINSKY_AH': True,
     'SMAG_BI_CONST': [i/100 for i in range(1,11)], 
     'USE_ZB2020': 'True',
     'amplitude': [i/24 for i in range(1,25)], 
     'ZB_type': 0, 
     'ZB_cons': 1, 
     'LPF_iter': 0, 
     'LPF_order': 1, 
     'HPF_iter': 0,
     'HPF_order': 1,
     'Stress_iter': [i for i in range(1,5)],
     'Stress_order': [i for i in range(1,5)]
}

run_many_experiments('/scratch/pp2681/mom6/Apr2022/R4/', hpc, parameters, max_runs=20)

# Perezhogin 2021 models
parameters = {'resolution': 'R4',
     'DAYMAX': 7300.0,
     'RESTINT': 1825.0,
     'LAPLACIAN': 'False',
     'BIHARMONIC': 'True',
     'SMAGORINSKY_AH': True,
     'SMAG_BI_CONST': [i/100 for i in range(1,11)], 
     'USE_ZB2020': 'True',
     'amplitude': [i/1. for i in range(1,25)], 
     'ZB_type': 0, 
     'ZB_cons': 1, 
     'LPF_iter': [i for i in range(1,5)], 
     'LPF_order': [i for i in range(1,5)], 
     'HPF_iter': [i for i in range(1,5)],
     'HPF_order': [i for i in range(1,5)],
     'Stress_iter': 0,
     'Stress_order': 1
}

run_many_experiments('/scratch/pp2681/mom6/Apr2022/R4/', hpc, parameters, max_runs=20)

# Alltogether models
parameters = {'resolution': 'R4',
     'DAYMAX': 7300.0,
     'RESTINT': 1825.0,
     'LAPLACIAN': 'False',
     'BIHARMONIC': 'True',
     'SMAGORINSKY_AH': True,
     'SMAG_BI_CONST': [i/100 for i in range(1,11)], 
     'USE_ZB2020': 'True',
     'amplitude': [i/1. for i in range(1,25)], 
     'ZB_type': 0, 
     'ZB_cons': 1, 
     'LPF_iter': [i for i in range(1,5)], 
     'LPF_order': [i for i in range(1,5)], 
     'HPF_iter': [i for i in range(1,5)],
     'HPF_order': [i for i in range(1,5)],
     'Stress_iter': [i for i in range(1,5)],
     'Stress_order': [i for i in range(1,5)]
}

run_many_experiments('/scratch/pp2681/mom6/Apr2022/R4/', hpc, parameters, max_runs=20)