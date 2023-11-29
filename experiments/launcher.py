import os
import json
import numpy as np

# creates slurm script mom.sub
def create_slurm(p, filename):
    # p - dictionary with parameters
    if p['mem'] < 1:
        mem = str(int(p['mem']*1000))+'MB'
    else:
        mem = str(p['mem'])+'GB'
    
    lines = [
    '#!/bin/bash',
    '#SBATCH --nodes='+str(p['nodes']),
    '#SBATCH --ntasks-per-node='+str(p['ntasks']),
    '#SBATCH --cpus-per-task=1',
    '#SBATCH --mem='+mem,
    '#SBATCH --time='+str(p['time'])+':00:00',
    '#SBATCH --begin=now+'+str(p['begin']),
    '#SBATCH --job-name='+str(p['name']),
    'scontrol show jobid -dd $SLURM_JOB_ID',
    'module purge',
    'source ~/MOM6-examples/build/intel/env',
    'time srun ./MOM6 > out.txt',
    'sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,Elapsed',
    'sacct -j $SLURM_JOB_ID --units=G --format=User,JobID%24,JobName,state,elapsed,TotalCPU,ReqMem,MaxRss,MaxVMSize,nnodes,ncpus,nodelist,Elapsed',
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
        print('Folder '+folder+' already exists. We skip it')
        return
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
            DT=2160.,
            DT_FORCING=2160.
        )
    if exp=='R3':
        return dictionary(
            NIGLOBAL=66,
            NJGLOBAL=60,
            DT=1440.,
            DT_FORCING=1440.
        )
    if exp=='R4':
        return dictionary(
            NIGLOBAL=88,
            NJGLOBAL=80,
            DT=1080.,
            DT_FORCING=1080.
        )
    if exp=='R5':
        return dictionary(
            NIGLOBAL=110,
            NJGLOBAL=100,
            DT=1080.,
            DT_FORCING=1080.
        )
    if exp=='R6':
        return dictionary(
            NIGLOBAL=132,
            NJGLOBAL=120,
            DT=720.,
            DT_FORCING=720.
        )
    if exp=='R7':
        return dictionary(
            NIGLOBAL=154,
            NJGLOBAL=140,
            DT=720.,
            DT_FORCING=720.
        )
    if exp=='R8':
        return dictionary(
            NIGLOBAL=176,
            NJGLOBAL=160,
            DT=540.,
            DT_FORCING=540.
        )
    if exp=='R10':
        return dictionary(
            NIGLOBAL=220,
            NJGLOBAL=200,
            DT=432.,
            DT_FORCING=432.
        )
    if exp=='R12':
        return dictionary(
            NIGLOBAL=264,
            NJGLOBAL=240,
            DT=360.,
            DT_FORCING=360.
        )
    if exp=='R16':
        return dictionary(
            NIGLOBAL=352,
            NJGLOBAL=320,
            DT=270.,
            DT_FORCING=270.
        )
    if exp=='R32':
        return dictionary(
            NIGLOBAL=704,
            NJGLOBAL=640,
            DT=135.,
            DT_FORCING=135.
        )
    if exp=='R64':
        return dictionary(
            NIGLOBAL=1408,
            NJGLOBAL=1280,
            DT=67.5,
            DT_FORCING=67.5
        )

HPC = dictionary(
    nodes=1,
    ntasks=1,
    mem=0.5,
    time=24,
    name='mom6',
    begin='0hour'
)  

PARAMETERS = dictionary(
    DAYMAX=7300.0,
    RESTINT=1825.0,
    LAPLACIAN='False',
    BIHARMONIC='True',
    SMAGORINSKY_AH='True',
    SMAG_BI_CONST=0.06, 
    USE_ZB2020='False',
    ZB_SCALING=1.,
    ZB_TRACE_MODE=0, 
    ZB_SCHEME=1, 
    VG_SHARP_PASS=0,
    VG_SHARP_SEL=1,
    STRESS_SMOOTH_PASS=0,
    STRESS_SMOOTH_SEL=1,
) + configuration('R4')

JansenHeld = dictionary(
    USE_MEKE='True',
    MEKE_VISCOSITY_COEFF_KU=-0.15, #See Jansen2019, top of page 2852
    RES_SCALE_MEKE_VISC=False, # Do not turn off parameterization if deformation radius is resolved
    MEKE_ADVECTION_FACTOR=1.0, # YES advection of MEKE
    MEKE_BACKSCAT_RO_POW=0.0, # Turn off Klower correction for MEKE source
    MEKE_USCALE=0.1, # velocity scale in bottom drag, see Eq. 9 in Jansen2019
    # MEKE_CDRAG is responsible for dissipation of MEKE near the bottom and automatically
    # will be chosen as 0.003, which is 10 smaller than the value in Jansen2019
    MEKE_GMCOEFF=-1.0, # No GM contribution
    MEKE_KHCOEFF=0.15, # Compute diffusivity from MEKE, with same parameter as for backscatter
    MEKE_FRCOEFF=1.0, # Conersion of dissipated KE to MEKE
    MEKE_KHMEKE_FAC=1.0, # diffusivity of MEKE is defined by the diffusivity coefficient
    MEKE_KH=0.0, # backgorund diffusivity of MEKE
    MEKE_CD_SCALE=1.0, # No intensification on the surface
    MEKE_CB=0.0,
    MEKE_CT=0.0,
    MEKE_MIN_LSCALE=True,
    MEKE_ALPHA_RHINES=1.0,
    MEKE_ALPHA_GRID=1.0,
    MEKE_COLD_START=True,
    MEKE_TOPOGRAPHIC_BETA=1.0,

    LAPLACIAN=True, # Allow Laplacian operator for backscattering
    KH=0.0, # No background diffusivity
    KH_VEL_SCALE=0.0, # No velocity scale to calculate diffusivity
    SMAGORINSKY_KH=False, # No Smagorinsky diffusivity
    BOUND_KH=False, # bounding is not needed for negative diffusivity
    BETTER_BOUND_KH=False
)

#########################################################################################
if __name__ == '__main__':
    # ################ old ANN experiments ####################
    # for conf in ['R4', 'R8']:#['R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']:
    #     for ZB_SCALING in [1.0]:#[1.0, 2.0]:
    #         for SMAG in [0.00]:#[0.00, 0.06]:
    #             parameters = PARAMETERS.add(USE_ZB2020='True',SMAG_BI_CONST=SMAG,ZB_SCALING=ZB_SCALING, USE_ANN=2).add(**configuration(conf))
    #             ntasks = dict(R2=4, R3=10, R4=24, R5=24, R6=24, R7=24, R8=24)[conf]
    #             hpc = HPC.add(mem=10, ntasks=ntasks)
    #             run_experiment(f'/scratch/pp2681/mom6/Apr2023/generalization/ANN_CM26_Kochkov_vorticity-{conf}/ZB-{ZB_SCALING}-Cs-{SMAG}', hpc, parameters)

    ################ new ANN experiments ####################
    # for conf in ['R4', 'R8']:#['R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']:
    #     for ZB_SCALING in [1.0]:#[1.0, 2.0]:
    #         for SMAG in [0.00]:#[0.00, 0.06]:
    #             for ann in ['hdn-20', 'hdn-64-64', 'hdn-64-64-sym', 'hdn-64-64-sym-trev']:
    #                 parameters = PARAMETERS.add(USE_ZB2020='True',SMAG_BI_CONST=SMAG,ZB_SCALING=ZB_SCALING, USE_ANN=2,
    #                     ANN_FILE_TXY=f'/scratch/pp2681/mom6/CM26_ML_models/Gauss-FGR2/{ann}/model/Txy_epoch_2000.nc',
    #                     ANN_FILE_TXX_TYY=f'/scratch/pp2681/mom6/CM26_ML_models/Gauss-FGR2/{ann}/model/Txx_Tyy_epoch_2000.nc').add(**configuration(conf))
    #                 ntasks = dict(R2=4, R3=10, R4=24, R5=24, R6=24, R7=24, R8=24)[conf]
    #                 hpc = HPC.add(mem=10, ntasks=ntasks)
    #                 run_experiment(f'/scratch/pp2681/mom6/CM26_Double_Gyre/generalization/{ann}-{conf}/ZB-{ZB_SCALING}-Cs-{SMAG}', hpc, parameters)

    ################# Backscatter is balanced with biharmonic Smagorinsky ##################
    # for conf in ['R4', 'R8']:#['R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']:
    #     for ZB_SCALING in [1.0]:#[1.0, 2.0]:
    #         for ann in ['hdn-20', 'hdn-64-64', 'hdn-64-64-sym-trev']:
    #             for STRESS_SMOOTH_PASS in [0, 1, 4]:
    #                 parameters = PARAMETERS.add(USE_ZB2020='True',
    #                     SMAG_BI_CONST=1., BOUND_CORIOLIS_BIHARM=False, BOUND_AH=False, BETTER_BOUND_AH=False, # These are needed for simple scaling
    #                     ZB_SCALING=ZB_SCALING, USE_ANN=2, ANN_SMAG_CONSERV=True, STRESS_SMOOTH_PASS=STRESS_SMOOTH_PASS,
    #                     ANN_FILE_TXY=f'/scratch/pp2681/mom6/CM26_ML_models/Gauss-FGR2/{ann}/model/Txy_epoch_2000.nc',
    #                     ANN_FILE_TXX_TYY=f'/scratch/pp2681/mom6/CM26_ML_models/Gauss-FGR2/{ann}/model/Txx_Tyy_epoch_2000.nc').add(**configuration(conf))
    #                 ntasks = dict(R2=4, R3=10, R4=24, R5=24, R6=24, R7=24, R8=24)[conf]
    #                 hpc = HPC.add(mem=10, ntasks=ntasks)
    #                 run_experiment(f'/scratch/pp2681/mom6/CM26_Double_Gyre/generalization/{ann}-{conf}/ZB-{ZB_SCALING}-Conserv-Smooth-{STRESS_SMOOTH_PASS}', hpc, parameters)

    # for conf in ['R4', 'R8']:#['R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']:
    #     for ZB_SCALING in [1.0]:#[1.0, 2.0]:
    #         parameters = PARAMETERS.add(USE_ZB2020='True',
    #             SMAG_BI_CONST=1., BOUND_CORIOLIS_BIHARM=False, BOUND_AH=False, BETTER_BOUND_AH=False, # These are needed for simple scaling
    #             ZB_SCALING=ZB_SCALING, USE_ANN=2, ANN_SMAG_CONSERV=True,
    #             ANN_FILE_TXY=f'/home/pp2681/MOM6-examples/src/MOM6/experiments/ANN-Results/trained_models/ANN_CM26_grid_harmonic_ver3/Txy_epoch_1000.nc',
    #             ANN_FILE_TXX_TYY=f'/home/pp2681/MOM6-examples/src/MOM6/experiments/ANN-Results/trained_models/ANN_CM26_grid_harmonic_ver3/Txx_Tyy_epoch_1000.nc').add(**configuration(conf))
    #         ntasks = dict(R2=4, R3=10, R4=24, R5=24, R6=24, R7=24, R8=24)[conf]
    #         hpc = HPC.add(mem=10, ntasks=ntasks)
    #         run_experiment(f'/scratch/pp2681/mom6/CM26_Double_Gyre/generalization/ANN_CM26_grid_harmonic_ver3-{conf}/ZB-{ZB_SCALING}-Conserv', hpc, parameters)

    ################# Backscatter is balanced by GM ##################
    for conf in ['R4', 'R8']:#['R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']:
        for ZB_SCALING in [1.0]:#[1.0, 2.0]:
            parameters = PARAMETERS.add(USE_ZB2020='True',
                SMAG_BI_CONST=0., BOUND_CORIOLIS_BIHARM=False, BOUND_AH=False, BETTER_BOUND_AH=False,
                ZB_SCALING=ZB_SCALING, USE_ANN=2, GM_CONSERV=True, THICKNESSDIFFUSE=True, KHTH=1,
                ANN_FILE_TXY=f'/home/pp2681/MOM6-examples/src/MOM6/experiments/ANN-Results/trained_models/ANN_CM26_grid_harmonic_ver3/Txy_epoch_1000.nc',
                ANN_FILE_TXX_TYY=f'/home/pp2681/MOM6-examples/src/MOM6/experiments/ANN-Results/trained_models/ANN_CM26_grid_harmonic_ver3/Txx_Tyy_epoch_1000.nc').add(**configuration(conf))
            ntasks = dict(R2=4, R3=10, R4=24, R5=24, R6=24, R7=24, R8=24)[conf]
            hpc = HPC.add(mem=10, ntasks=ntasks)
            run_experiment(f'/scratch/pp2681/mom6/CM26_Double_Gyre/generalization/ANN_CM26_grid_harmonic_ver3-{conf}/ZB-{ZB_SCALING}-GM-Conserv', hpc, parameters)

    for conf in ['R4', 'R8']:#['R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']:
        for ZB_SCALING in [1.0]:#[1.0, 2.0]:
            parameters = PARAMETERS.add(USE_ZB2020='True',
                SMAG_BI_CONST=0.,
                ZB_SCALING=ZB_SCALING, USE_ANN=2, GM_CONSERV=True, THICKNESSDIFFUSE=True, KHTH=1,
                ANN_FILE_TXY=f'/home/pp2681/MOM6-examples/src/MOM6/experiments/ANN-Results/trained_models/ANN_CM26_grid_harmonic_ver3/Txy_epoch_1000.nc',
                ANN_FILE_TXX_TYY=f'/home/pp2681/MOM6-examples/src/MOM6/experiments/ANN-Results/trained_models/ANN_CM26_grid_harmonic_ver3/Txx_Tyy_epoch_1000.nc').add(**configuration(conf))
            ntasks = dict(R2=4, R3=10, R4=24, R5=24, R6=24, R7=24, R8=24)[conf]
            hpc = HPC.add(mem=10, ntasks=ntasks)
            run_experiment(f'/scratch/pp2681/mom6/CM26_Double_Gyre/generalization/ANN_CM26_grid_harmonic_ver3-{conf}/ZB-{ZB_SCALING}-GM-Conserv-bound', hpc, parameters)