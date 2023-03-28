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
    '#SBATCH --begin=now+'+str(p['begin']),
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
    if exp=='R6':
        return dictionary(
            NIGLOBAL=132,
            NJGLOBAL=120,
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
    ntasks=4,
    mem=10,
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
    SMAG_BI_CONST=0.03, 
    USE_ZB2020='False',
    amplitude=1.,
    amp_bottom=-1.,
    ZB_type=0, 
    ZB_cons=1, 
    LPF_iter=0, 
    LPF_order=1, 
    HPF_iter=0,
    HPF_order=1,
    Stress_iter=0,
    Stress_order=1,
    ssd_iter=-1,
    ssd_bound_coef=0.8
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
    # for ntasks in [1, 4, 8, 10, 16, 24, 48]:
    #     run_experiment(f'/scratch/pp2681/mom6/Apr2022/runtime/R4-ntasks-{ntasks}', HPC.add(ntasks=ntasks, time=1), PARAMETERS.add(DAYMAX=100))

    # parameters = PARAMETERS.add(USE_ZB2020='True')
    # for amp in range(0,11):
    #     for DT in [270, 1080, 4320]:
    #         parameters = parameters.add(amplitude=amp/24., DT=DT)
    #         run_experiment(f'/scratch/pp2681/mom6/Apr2022/R4/Smagorinsky-ZB-dt/Cs-0.03-ZB-{amp}-24-DT-{DT}', HPC, parameters)

    # parameters = PARAMETERS.add(USE_ZB2020='True')
    # for amplitude in [0,2,4,6,8,10]:
    #     for amp_bottom in [0,2,4,6,8,10]:
    #         parameters = parameters.add(amplitude=amplitude/24., amp_bottom=amp_bottom/24.)
    #         run_experiment(f'/scratch/pp2681/mom6/Apr2022/R4/Smagorinsky-ZB-layers/upper-{amplitude}-24-lower-{amp_bottom}-24', HPC, parameters)

    # for Stress_iter, Stress_order in [(1,1), (2,1), (4,1), (1,2), (1,4), (1,8), (2,2), (4,4)]:
    #     parameters = PARAMETERS.add(USE_ZB2020='True', Stress_iter=Stress_iter,Stress_order=Stress_order)
    #     for amplitude in [0,2,4,6,8,10]:
    #         for amp_bottom in [0,2,4,6,8,10]:
    #             parameters = parameters.add(amplitude=amplitude/10., amp_bottom=amp_bottom/10.)
    #             run_experiment(f'/scratch/pp2681/mom6/Apr2022/R4/Smagorinsky-ZB-filters/stress_iter-{Stress_iter}-stress_order-{Stress_order}-upper-{amplitude}-10-lower-{amp_bottom}-10', HPC.add(ntasks=2), parameters)

    # parameters = PARAMETERS.add(USE_ZB2020='True', Stress_iter=1,Stress_order=2,HPF_iter=1,HPF_order=1)
    # for amplitude in [0,2,4,6,8,10]:
    #     for amp_bottom in [0,2,4,6,8,10]:
    #         parameters = parameters.add(amplitude=amplitude/5., amp_bottom=amp_bottom/5.)
    #         run_experiment(f'/scratch/pp2681/mom6/Apr2022/R4/Smagorinsky-ZB-HPF/upper-{amplitude}-5-lower-{amp_bottom}-5', HPC.add(ntasks=2), parameters)

    # parameters = PARAMETERS.add(USE_ZB2020='True', Stress_iter=1,Stress_order=2,HPF_iter=1,HPF_order=1,LPF_iter=1,LPF_order=2)
    # for amplitude in [0,2,4,6,8,10]:
    #     for amp_bottom in [0,2,4,6,8,10]:
    #         parameters = parameters.add(amplitude=amplitude/3., amp_bottom=amp_bottom/3.)
    #         run_experiment(f'/scratch/pp2681/mom6/Apr2022/R4/Smagorinsky-ZB-HPF-LPF/upper-{amplitude}-3-lower-{amp_bottom}-3', HPC.add(ntasks=2), parameters)

    # for j, amplitude in enumerate(np.linspace(0,10,41)):
    #     parameters = PARAMETERS.add(USE_ZB2020='True',Stress_iter=4,Stress_order=1,amplitude=amplitude)
    #     run_experiment(f'/scratch/pp2681/mom6/Apr2022/R4/Smagorinsky-ZB-1D/momentum-4-1/EXP{j}', HPC.add(ntasks=2), parameters)

    # for j, amplitude in enumerate(np.linspace(0,10,41)):
    #     parameters = PARAMETERS.add(USE_ZB2020='True',Stress_iter=1,Stress_order=2,HPF_iter=1,HPF_order=1,amplitude=amplitude)
    #     run_experiment(f'/scratch/pp2681/mom6/Apr2022/R4/Smagorinsky-ZB-1D/HPF/EXP{j}', HPC.add(ntasks=2), parameters)

    # for j, amplitude in enumerate(np.linspace(0,10,41)):
    #     parameters = PARAMETERS.add(USE_ZB2020='True',Stress_iter=1,Stress_order=2,HPF_iter=1,HPF_order=1,LPF_iter=1,LPF_order=2,amplitude=amplitude)
    #     run_experiment(f'/scratch/pp2681/mom6/Apr2022/R4/Smagorinsky-ZB-1D/HPF-LPF/EXP{j}', HPC.add(ntasks=2), parameters)

    # for j, amplitude in enumerate(np.linspace(0,10,41)):
    #     parameters = PARAMETERS.add(USE_ZB2020='True',Stress_iter=1,Stress_order=2,HPF_iter=2,HPF_order=2,LPF_iter=1,LPF_order=4,amplitude=amplitude)
    #     run_experiment(f'/scratch/pp2681/mom6/Apr2022/R4/Smagorinsky-ZB-1D/best/EXP{j}', HPC.add(ntasks=10), parameters)

    # for j, amplitude in enumerate(np.linspace(0,10,41)):
    #     parameters = PARAMETERS.add(USE_ZB2020='True',Stress_iter=1,Stress_order=2,HPF_iter=2,HPF_order=2,amplitude=amplitude)
    #     run_experiment(f'/scratch/pp2681/mom6/Apr2022/R4/Smagorinsky-ZB-1D/best-noLPF/EXP{j}', HPC.add(ntasks=10), parameters)

    # for j, amplitude in enumerate(np.linspace(0,10,41)):
    #     parameters = PARAMETERS.add(USE_ZB2020='True',HPF_iter=2,HPF_order=2,LPF_iter=1,LPF_order=4,amplitude=amplitude)
    #     run_experiment(f'/scratch/pp2681/mom6/Apr2022/R4/Smagorinsky-ZB-1D/best-noStress/EXP{j}', HPC.add(ntasks=10), parameters)

    # for j, amplitude in enumerate(np.linspace(0,10,41)):
    #     parameters = PARAMETERS.add(USE_ZB2020='True',HPF_iter=2,HPF_order=2,amplitude=amplitude)
    #     run_experiment(f'/scratch/pp2681/mom6/Apr2022/R4/Smagorinsky-ZB-1D/best-HPFonly/EXP{j}', HPC.add(ntasks=10), parameters)

    # for conf in ['R10', 'R12']:#['R2', 'R3', 'R4', 'R6', 'R8', 'R10', 'R12', 'R16']:
    #     ntasks = dict(R2=4, R3=4, R4=16, R6=48, R8=48, R10=48, R12=48, R16=48)[conf]
    #     nodes = dict(R2=1, R3=1, R4=1, R6=1, R8=1, R10=2, R12=2, R16=3)[conf]
    #     begin = '0hour' if conf in ['R2', 'R3', 'R4', 'R6', 'R8'] else '0hour'
    #     parameters0 = PARAMETERS.add(**configuration(conf))
    #     run_experiment(f'/scratch/pp2681/mom6/Apr2022/generalization/{conf}_ref', HPC.add(ntasks=ntasks,nodes=nodes,mem=32,begin=begin,name=conf+'-ref'), parameters0)
    #     parameters = parameters0.add(USE_ZB2020='True',Stress_iter=1,Stress_order=2,HPF_iter=2,HPF_order=2,LPF_iter=1,LPF_order=4,amplitude=7.0)
    #     #run_experiment(f'/scratch/pp2681/mom6/Apr2022/generalization/{conf}_EXP205', HPC.add(ntasks=ntasks,ndoes=nodes,mem=32,begin=begin,name=conf+'-205'), parameters)
    #     parameters = parameters0.add(USE_ZB2020='True',Stress_iter=4,Stress_order=1,amplitude=0.75)
    #     run_experiment(f'/scratch/pp2681/mom6/Apr2022/generalization/{conf}_momf', HPC.add(ntasks=ntasks,nodes=nodes,mem=32,begin=begin,name=conf+'-mom'), parameters)
    #     print(conf+' done')
    #     input()

    # parameters = PARAMETERS.add(USE_ZB2020='True',amplitude=0., ssd_iter=4, ssd_bound_coef=0.05)
    # run_experiment(f'/scratch/pp2681/mom6/Apr2022/gov-eq-filter/R4-filter-4-0.05', HPC.add(ntasks=14), parameters)

    # parameters = PARAMETERS.add(USE_ZB2020='True',amplitude=0., ssd_iter=10, ssd_bound_coef=0.05)
    # run_experiment(f'/scratch/pp2681/mom6/Apr2022/gov-eq-filter/R4-filter-10-0.05', HPC.add(ntasks=14), parameters)

    # for amplitude in [i/10. for i in range(0,11)]:
    #     parameters = PARAMETERS.add(USE_ZB2020='True',amplitude=amplitude, ssd_iter=4, ssd_bound_coef=0.05)
    #     run_experiment(f'/scratch/pp2681/mom6/Apr2022/gov-eq-filter/ZB-ssd-4-iter/amplitude-{str(amplitude)}', HPC.add(ntasks=10), parameters)

    # for amplitude in [i/10. for i in range(0,11)]:
    #     parameters = PARAMETERS.add(USE_ZB2020='True',amplitude=amplitude, ssd_iter=10, ssd_bound_coef=0.05)
    #     run_experiment(f'/scratch/pp2681/mom6/Apr2022/gov-eq-filter/ZB-ssd-10-iter/amplitude-{str(amplitude)}', HPC.add(ntasks=10), parameters)

    # for amplitude in [i/10. for i in range(0,11)]:
    #     parameters = PARAMETERS.add(USE_ZB2020='True',amplitude=amplitude, ssd_iter=10, ssd_bound_coef=0.2)
    #     run_experiment(f'/scratch/pp2681/mom6/Apr2022/gov-eq-filter/ZB-ssd-10-iter-0.2/amplitude-{str(amplitude)}', HPC.add(ntasks=10), parameters)

    # for amplitude in [i/10. for i in range(0,11)]:
    #     parameters = PARAMETERS.add(USE_ZB2020='True',amplitude=amplitude, ssd_iter=10, ssd_bound_coef=0.8)
    #     run_experiment(f'/scratch/pp2681/mom6/Apr2022/gov-eq-filter/ZB-ssd-10-iter-0.8/amplitude-{str(amplitude)}', HPC.add(ntasks=10), parameters)

    # for conf in ['R2', 'R3', 'R4', 'R6', 'R8']:
    #     ntasks = dict(R2=4, R3=4, R4=16, R6=24, R8=24, R10=48, R12=48, R16=48)[conf]
    #     nodes = dict(R2=1, R3=1, R4=1, R6=1, R8=1, R10=2, R12=2, R16=3)[conf]
    #     parameters0 = PARAMETERS.add(**configuration(conf))
    #     parameters = parameters0.add(USE_ZB2020='True',amplitude=0.3)
    #     run_experiment(f'/scratch/pp2681/mom6/Apr2022/generalization/{conf}_ZB', HPC.add(ntasks=ntasks,nodes=nodes,mem=16,name=conf+'-ZB'), parameters)
    #     parameters = parameters0.add(USE_ZB2020='True',amplitude=0.3,ssd_iter=10,ssd_bound_coef=0.2)
    #     run_experiment(f'/scratch/pp2681/mom6/Apr2022/generalization/{conf}_ZB-ssd', HPC.add(ntasks=ntasks,nodes=nodes,mem=16,name=conf+'-ZBsd'), parameters)
    #     print(conf+' done')
    #     input()

    # for ZB_type, postfix in zip([0,1,2],['full','trace-free','trace-only']):
    #     parameters = PARAMETERS.add(USE_ZB2020='True',Stress_iter=1,Stress_order=2,HPF_iter=2,HPF_order=2,LPF_iter=1,LPF_order=4,amplitude=7.0, ZB_type=ZB_type)
    #     run_experiment(f'/scratch/pp2681/mom6/Apr2022/trace/EXP205-{postfix}', HPC, parameters)
    #     parameters = PARAMETERS.add(USE_ZB2020='True',Stress_iter=4,Stress_order=1,amplitude=0.75, ZB_type=ZB_type)
    #     run_experiment(f'/scratch/pp2681/mom6/Apr2022/trace/momf-{postfix}', HPC, parameters)
    #     parameters = PARAMETERS.add(USE_ZB2020='True',amplitude=0.3, ZB_type=ZB_type)
    #     run_experiment(f'/scratch/pp2681/mom6/Apr2022/trace/ZB-{postfix}', HPC, parameters)
    #     parameters = PARAMETERS.add(USE_ZB2020='True',amplitude=0.3,ssd_iter=10,ssd_bound_coef=0.2, ZB_type=ZB_type)
    #     run_experiment(f'/scratch/pp2681/mom6/Apr2022/trace/ZB-ssd-{postfix}', HPC, parameters)

    run_experiment(f'/scratch/pp2681/mom6/Apr2022/R4/JansenHeld/R4-JH', HPC, PARAMETERS+JansenHeld)