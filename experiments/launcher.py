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
    'scontrol show jobid -dd $SLURM_JOB_ID',
    'module purge',
    'source ~/MOM6-examples/build/intel/env',
    'time mpiexec ./MOM6 > out.txt',
    'sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,Elapsed',
    'mkdir -p output',
    'mv *.nc output'
    ]
    with open(filename,'w') as fid:
        fid.writelines([ line+'\n' for line in lines])

def create_MOM_override(p, filename):
    # p - dictionary of parameters
    lines = []
    for key in p.keys():
        lines.append('#override '+key+' = '+str(p[key]))
    with open(filename,'w') as fid:
        fid.writelines([ line+'\n' for line in lines])

def run_experiment(folder, hpc, parameters):
    if os.path.exists(folder):
        print('Folder '+folder+' already exists. Delete it? (y/n)')
        if input()!='y':
            print('Experiment is not launched. Exit.')
            return
        else:
            os.system('rm -r '+folder)
    os.system('mkdir -p '+folder)
    
    create_slurm(hpc, os.path.join(folder,'mom.sub'))
    create_MOM_override(parameters, os.path.join(folder,'MOM_override'))
    
    os.system('cp -r ~/MOM6-examples/src/MOM6/experiments/configurations/double_gyre/* '+folder)
    os.system('cp ~/MOM6-examples/build/intel/ocean_only/repro/MOM6 '+folder)

    with open(os.path.join(folder,'args.json'), 'w') as f:
        json.dump(parameters, f, indent=2)
    
    os.system('cd '+folder+'; sbatch mom.sub')

#########################################################################################
class dictionary(dict):  
    def __init__(self, **kw):  
        super().__init__(**kw)
    def add(self, **kw): 
        d = self.copy()
        d.update(kw)
        return dictionary(**d)
    def __add__(self, d):
        return self.add(**d)
    
def configuration(exp='R4'):
    if exp=='R2':
        return dictionary(
            NIGLOBAL=44,
            NJGLOBAL=40,
            DT=2160.
        )
    if exp=='R3':
        return dictionary(
            NIGLOBAL=66,
            NJGLOBAL=60,
            DT=1440.
        )
    if exp=='R4':
        return dictionary(
            NIGLOBAL=88,
            NJGLOBAL=80,
            DT=1080.
        )
    if exp=='R6':
        return dictionary(
            NIGLOBAL=132,
            NJGLOBAL=120,
            DT=720.
        )
    if exp=='R8':
        return dictionary(
            NIGLOBAL=176,
            NJGLOBAL=160,
            DT=540.
        )
    if exp=='R10':
        return dictionary(
            NIGLOBAL=220,
            NJGLOBAL=200,
            DT=432.
        )
    if exp=='R12':
        return dictionary(
            NIGLOBAL=264,
            NJGLOBAL=240,
            DT=360.
        )
    if exp=='R16':
        return dictionary(
            NIGLOBAL=352,
            NJGLOBAL=320,
            DT=270.
        )
    if exp=='R32':
        return dictionary(
            NIGLOBAL=704,
            NJGLOBAL=640,
            DT=135.
        )
    if exp=='R64':
        return dictionary(
            NIGLOBAL=1408,
            NJGLOBAL=1280,
            DT=67.5
        )

HPC = dictionary(
    nodes=1,
    ntasks=4,
    mem=10,
    time=24,
    name='mom6'
)  

PARAMETERS = dictionary(
    DAYMAX=7300.0,
    RESTINT=1825.0,
    LAPLACIAN='False',
    BIHARMONIC='True',
    SMAGORINSKY_AH='True',
    SMAG_BI_CONST=0.03, 
    USE_ZB2020='False',
    amplitude=1., 
    ZB_type=0, 
    ZB_cons=1, 
    LPF_iter=0, 
    LPF_order=1, 
    HPF_iter=0,
    HPF_order=1,
    Stress_iter=0,
    Stress_order=1
) + configuration('R4')

#########################################################################################
if __name__ == '__main__':
    # for ntasks in [1, 4, 8, 10, 16, 24, 48]:
    #     run_experiment(f'/scratch/pp2681/mom6/Apr2022/runtime/R4-ntasks-{ntasks}', HPC.add(ntasks=ntasks, time=1), PARAMETERS.add(DAYMAX=100))

    parameters = PARAMETERS.add(USE_ZB2020='True')
    for amp in range(0,11):
        for DT in [270, 1080, 4320]:
            parameters = parameters.add(amplitude=amp/24., DT=DT)
            run_experiment(f'/scratch/pp2681/mom6/Apr2022/R4/Smagorinsky-ZB-dt/Cs-0.03-ZB-{amp}-24-DT-{DT}', HPC, parameters)