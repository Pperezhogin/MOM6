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
    'time mpiexec ./MOM6 > out.txt',
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
    SMAG_BI_CONST=0.03, 
    USE_ZB2020='False',
    ZB_SCALING=1.,
    ZB_TRACE_MODE=0, 
    ZB_SCHEME=1, 
    VG_SMOOTH_PASS=0, 
    VG_SMOOTH_SEL=1, 
    VG_SHARP_PASS=0,
    VG_SHARP_SEL=1,
    STRESS_SMOOTH_PASS=0,
    STRESS_SMOOTH_SEL=1,
    ZB_HYPERVISC=0,
    HYPVISC_GRID_DAMP=0.2
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

    #run_experiment(f'/scratch/pp2681/mom6/Apr2022/R4/JansenHeld/R4-JH', HPC, PARAMETERS+JansenHeld)

    ####################################### Hopefully final runs ########################################

    # for ZB_SCALING in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
    #     for Cs in ['0.00']:#[0.03, 0.06, 0.09]:
    #         parameters = PARAMETERS.add(USE_ZB2020='True',SMAG_BI_CONST=Cs,ZB_SCALING=ZB_SCALING)
    #         run_experiment(f'/scratch/pp2681/mom6/Apr2023/R4-sensitivity/ZB-clean/Cs-{Cs}-ZB-{ZB_SCALING}', HPC, parameters)
        
    # for ZB_SCALING in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
    #     for Cs in [0.03, 0.06, 0.09]:
    #         parameters = PARAMETERS.add(USE_ZB2020='True',SMAG_BI_CONST=Cs,ZB_SCALING=ZB_SCALING, ZB_HYPERVISC=10)
    #         run_experiment(f'/scratch/pp2681/mom6/Apr2023/R4-sensitivity/ZB-ssd/Cs-{Cs}-ZB-{ZB_SCALING}', HPC, parameters)

    # for ZB_SCALING in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
    #     for Cs in ['0.00', '0.03', '0.06', '0.09']:
    #         parameters = PARAMETERS.add(USE_ZB2020='True',SMAG_BI_CONST=Cs,ZB_SCALING=ZB_SCALING, ZB_HYPERVISC=5,HYPVISC_GRID_DAMP=0.05)
    #         run_experiment(f'/scratch/pp2681/mom6/Apr2023/R4-sensitivity/ZB-ssd-5/Cs-{Cs}-ZB-{ZB_SCALING}', HPC, parameters)

    # for STRESS_SMOOTH_PASS, STRESS_SMOOTH_SEL in [(4,1)]:#[(1,1), (2,1), (3,1), (4,1), (1,2), (1,4), (2,2), (4,4)]:
    #     for ZB_SCALING in [1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]: #[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
    #         for Cs in [0.03, 0.06, 0.09]:
    #             parameters = PARAMETERS.add(USE_ZB2020='True',SMAG_BI_CONST=Cs,ZB_SCALING=ZB_SCALING,STRESS_SMOOTH_PASS=STRESS_SMOOTH_PASS,STRESS_SMOOTH_SEL=STRESS_SMOOTH_SEL)
    #             run_experiment(f'/scratch/pp2681/mom6/Apr2023/R4-sensitivity/ZB-stress-pass-{STRESS_SMOOTH_PASS}-sel-{STRESS_SMOOTH_SEL}/Cs-{Cs}-ZB-{ZB_SCALING}', HPC, parameters)
        
        # for ZB_SCALING in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]:
        #     for Cs in ['0.00']:
        #         parameters = PARAMETERS.add(USE_ZB2020='True',SMAG_BI_CONST=Cs,ZB_SCALING=ZB_SCALING,STRESS_SMOOTH_PASS=STRESS_SMOOTH_PASS,STRESS_SMOOTH_SEL=STRESS_SMOOTH_SEL)
        #         run_experiment(f'/scratch/pp2681/mom6/Apr2023/R4-sensitivity/ZB-stress-pass-{STRESS_SMOOTH_PASS}-sel-{STRESS_SMOOTH_SEL}/Cs-{Cs}-ZB-{ZB_SCALING}', HPC, parameters)

######################################################## non-simple experiments ########################################################

    # for ZB_SCALING in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
    #     for Cs in [0.03, 0.06, 0.09]:
    #         parameters = PARAMETERS.add(USE_ZB2020='True',SMAG_BI_CONST=Cs,ZB_SCALING=ZB_SCALING, ZB_HYPERVISC=11)
    #         run_experiment(f'/scratch/pp2681/mom6/Apr2023/R4-sensitivity/ZB-ssd-11/Cs-{Cs}-ZB-{ZB_SCALING}', HPC, parameters)

    # for ZB_SCALING in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
    #     for Cs in [0.03, 0.06, 0.09]:
    #         parameters = PARAMETERS.add(USE_ZB2020='True',SMAG_BI_CONST=Cs,ZB_SCALING=ZB_SCALING,VG_SMOOTH_PASS=2)
    #         run_experiment(f'/scratch/pp2681/mom6/Apr2023/R4-sensitivity/ZB-VG-pass-2/Cs-{Cs}-ZB-{ZB_SCALING}', HPC, parameters)

    # for SMOOTH_PASS, SMOOTH_SEL in [(1,1), (2,1), (4,1), (1,2), (1,4), (2,2), (4,4)]:
    #     for ZB_SCALING in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]:
    #         for Cs in [0.03, 0.06, 0.09]:
    #             parameters = PARAMETERS.add(USE_ZB2020='True',SMAG_BI_CONST=Cs,ZB_SCALING=ZB_SCALING,STRESS_SMOOTH_PASS=SMOOTH_PASS,STRESS_SMOOTH_SEL=SMOOTH_SEL,VG_SHARP_PASS=SMOOTH_PASS,VG_SHARP_SEL=SMOOTH_SEL)
    #             run_experiment(f'/scratch/pp2681/mom6/Apr2023/R4-change-range/ZB-Reynolds-pass-{SMOOTH_PASS}-sel-{SMOOTH_SEL}/Cs-{Cs}-ZB-{ZB_SCALING}', HPC, parameters)

    # for SMOOTH_PASS, SMOOTH_SEL in [(1,1), (2,1), (4,1), (1,2), (1,4), (2,2), (4,4)]:
    #     for ZB_SCALING in [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0]:
    #         for Cs in [0.03, 0.06, 0.09]:
    #             parameters = PARAMETERS.add(USE_ZB2020='True',SMAG_BI_CONST=Cs,ZB_SCALING=ZB_SCALING,STRESS_SMOOTH_PASS=SMOOTH_PASS,STRESS_SMOOTH_SEL=SMOOTH_SEL,VG_SHARP_PASS=SMOOTH_PASS,VG_SHARP_SEL=SMOOTH_SEL,VG_SMOOTH_PASS=SMOOTH_PASS,VG_SMOOTH_SEL=SMOOTH_SEL)
    #             run_experiment(f'/scratch/pp2681/mom6/Apr2023/R4-change-range/ZB-Reynolds-VG-smooth-pass-{SMOOTH_PASS}-sel-{SMOOTH_SEL}/Cs-{Cs}-ZB-{ZB_SCALING}', HPC, parameters)

    # for ZB_SCALING in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]:
    #     for Cs in [0.03, 0.06, 0.09]:
    #         parameters = PARAMETERS.add(USE_ZB2020='True',SMAG_BI_CONST=Cs,ZB_SCALING=ZB_SCALING,STRESS_SMOOTH_PASS=1,STRESS_SMOOTH_SEL=2,VG_SHARP_PASS=2,VG_SHARP_SEL=2,VG_SMOOTH_PASS=1,VG_SMOOTH_SEL=4)
    #         run_experiment(f'/scratch/pp2681/mom6/Apr2023/R4-sensitivity/ZB-EXP205/Cs-{Cs}-ZB-{ZB_SCALING}', HPC, parameters)

    # for ZB_SCALING in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0]:
    #     for Cs in ['0.00']:#[0.03, 0.06, 0.09]:
    #         parameters = PARAMETERS.add(USE_ZB2020='True',SMAG_BI_CONST=Cs,ZB_SCALING=ZB_SCALING,STRESS_SMOOTH_PASS=4,STRESS_SMOOTH_SEL=1,VG_SHARP_PASS=4,VG_SHARP_SEL=1)
    #         run_experiment(f'/scratch/pp2681/mom6/Apr2023/R4-selected/ZB-Reynolds-pass-4-pass-4/Cs-{Cs}-ZB-{ZB_SCALING}', HPC, parameters)

    # for ZB_SCALING in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0]:
    #     for Cs in [0.03, 0.06, 0.09]:
    #         parameters = PARAMETERS.add(USE_ZB2020='True',SMAG_BI_CONST=Cs,ZB_SCALING=ZB_SCALING,STRESS_SMOOTH_PASS=4,STRESS_SMOOTH_SEL=1,VG_SHARP_PASS=1,VG_SHARP_SEL=1)
    #         run_experiment(f'/scratch/pp2681/mom6/Apr2023/R4-selected/ZB-Reynolds-pass-4-pass-1/Cs-{Cs}-ZB-{ZB_SCALING}', HPC, parameters)

    # for ZB_SCALING in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0]:
    #     for Cs in [0.03, 0.06, 0.09]:
    #         parameters = PARAMETERS.add(USE_ZB2020='True',SMAG_BI_CONST=Cs,ZB_SCALING=ZB_SCALING,STRESS_SMOOTH_PASS=4,STRESS_SMOOTH_SEL=1,VG_SHARP_PASS=2,VG_SHARP_SEL=1)
    #         run_experiment(f'/scratch/pp2681/mom6/Apr2023/R4-selected/ZB-Reynolds-pass-4-pass-2/Cs-{Cs}-ZB-{ZB_SCALING}', HPC, parameters)

    # for ZB_SCALING in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0]:
    #     for Cs in [0.03, 0.06, 0.09]:
    #         parameters = PARAMETERS.add(USE_ZB2020='True',SMAG_BI_CONST=Cs,ZB_SCALING=ZB_SCALING,STRESS_SMOOTH_PASS=4,STRESS_SMOOTH_SEL=1,VG_SHARP_PASS=1,VG_SHARP_SEL=2)
    #         run_experiment(f'/scratch/pp2681/mom6/Apr2023/R4-selected/ZB-Reynolds-pass-4-pass-1-sel-2/Cs-{Cs}-ZB-{ZB_SCALING}', HPC, parameters)


    ############################################ generalization experiments ################################################
    for conf in ['R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']:
        for ZB_SCALING in [1.5, 1.6, 2.5]:#[2.0]:#[2.4, 2.6]:#[0.8, 1.2, 1.8, 2.2, 2.8]:
            parameters = PARAMETERS.add(USE_ZB2020='True',SMAG_BI_CONST=0.06,ZB_SCALING=ZB_SCALING,STRESS_SMOOTH_PASS=4,STRESS_SMOOTH_SEL=1,VG_SHARP_PASS=4,VG_SHARP_SEL=1).add(**configuration(conf))
            ntasks = dict(R2=1, R3=1, R4=1, R5=1, R6=10, R7=10, R8=10)[conf]
            hpc = HPC.add(mem=4, ntasks=ntasks)
            run_experiment(f'/scratch/pp2681/mom6/Apr2023/generalization/ZB-Reynolds-{conf}/ZB-{ZB_SCALING}', hpc, parameters)

    for conf in ['R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']:
        for ZB_SCALING in [1.5, 1.6, 2.0, 2.4, 2.5]:#[0.8, 1.2, 1.8, 2.2, 2.8]:
            parameters = PARAMETERS.add(USE_ZB2020='True',SMAG_BI_CONST=0.06,ZB_SCALING=ZB_SCALING,STRESS_SMOOTH_PASS=2,STRESS_SMOOTH_SEL=1,VG_SHARP_PASS=2,VG_SHARP_SEL=1).add(**configuration(conf))
            ntasks = dict(R2=1, R3=1, R4=1, R5=1, R6=10, R7=10, R8=10)[conf]
            hpc = HPC.add(mem=4, ntasks=ntasks)
            run_experiment(f'/scratch/pp2681/mom6/Apr2023/generalization/ZB-Reynolds-2-{conf}/ZB-{ZB_SCALING}', hpc, parameters)
    
    for conf in ['R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']:
        for ZB_SCALING in [1.0]:#[0.5, 0.7, 0.9, 1.1, 1.3, 1.5]:
            parameters = PARAMETERS.add(USE_ZB2020='True',SMAG_BI_CONST=0.06,ZB_SCALING=ZB_SCALING,STRESS_SMOOTH_PASS=4,STRESS_SMOOTH_SEL=1).add(**configuration(conf))
            ntasks = dict(R2=1, R3=1, R4=1, R5=1, R6=10, R7=10, R8=10)[conf]
            hpc = HPC.add(mem=4, ntasks=ntasks)
            run_experiment(f'/scratch/pp2681/mom6/Apr2023/generalization/ZB-Smooth-{conf}/ZB-{ZB_SCALING}', hpc, parameters)

    for conf in ['R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']:
        for ZB_SCALING in [1.0]:#[0.5, 0.7, 0.9, 1.1, 1.3, 1.5]:
            parameters = PARAMETERS.add(USE_ZB2020='True',SMAG_BI_CONST=0.06,ZB_SCALING=ZB_SCALING,STRESS_SMOOTH_PASS=2,STRESS_SMOOTH_SEL=1).add(**configuration(conf))
            ntasks = dict(R2=1, R3=1, R4=1, R5=1, R6=10, R7=10, R8=10)[conf]
            hpc = HPC.add(mem=4, ntasks=ntasks)
            run_experiment(f'/scratch/pp2681/mom6/Apr2023/generalization/ZB-Smooth-2-{conf}/ZB-{ZB_SCALING}', hpc, parameters)

    # for conf in ['R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']:
    #     for ZB_SCALING in [0.3, 0.4, 0.5]:
    #         parameters = PARAMETERS.add(USE_ZB2020='True',SMAG_BI_CONST=0.06,ZB_SCALING=ZB_SCALING).add(**configuration(conf))
    #         ntasks = dict(R2=1, R3=1, R4=1, R5=1, R6=10, R7=10, R8=10)[conf]
    #         hpc = HPC.add(mem=4, ntasks=ntasks)
    #         run_experiment(f'/scratch/pp2681/mom6/Apr2023/generalization/ZB-clean-{conf}/ZB-{ZB_SCALING}', hpc, parameters)

    # for conf in ['R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']:
    #     for ZB_SCALING in [0.3, 0.4, 0.5, 0.6, 0.7]:
    #         parameters = PARAMETERS.add(USE_ZB2020='True',SMAG_BI_CONST=0.06,ZB_SCALING=ZB_SCALING,ZB_HYPERVISC=11).add(**configuration(conf))
    #         ntasks = dict(R2=1, R3=1, R4=1, R5=1, R6=10, R7=10, R8=10)[conf]
    #         hpc = HPC.add(mem=4, ntasks=ntasks)
    #         run_experiment(f'/scratch/pp2681/mom6/Apr2023/generalization/ZB-ssd-{conf}/ZB-{ZB_SCALING}', hpc, parameters)

    # for conf in ['R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']:
    #     for ZB_SCALING in [0.3, 0.4, 0.5, 0.6, 0.7]:
    #         parameters = PARAMETERS.add(USE_ZB2020='True',SMAG_BI_CONST=0.06,ZB_SCALING=ZB_SCALING,ZB_HYPERVISC=5,HYPVISC_GRID_DAMP=0.05).add(**configuration(conf))
    #         ntasks = dict(R2=1, R3=1, R4=1, R5=1, R6=10, R7=10, R8=10)[conf]
    #         hpc = HPC.add(mem=4, ntasks=ntasks)
    #         run_experiment(f'/scratch/pp2681/mom6/Apr2023/generalization/ZB-ssd-5-{conf}/ZB-{ZB_SCALING}', hpc, parameters)

    # for conf in ['R5', 'R6', 'R7']:
    #     parameters = PARAMETERS.add(SMAG_BI_CONST=0.06).add(**configuration(conf))
    #     ntasks = dict(R2=1, R3=1, R4=1, R5=1, R6=10, R7=10, R8=10)[conf]
    #     hpc = HPC.add(mem=4, ntasks=ntasks)
    #     run_experiment(f'/scratch/pp2681/mom6/Apr2023/generalization/Smag-{conf}/Cs-0.06', hpc, parameters)


    # for conf in ['R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']:
    #     for ZB_SCALING in [3.0, 7.0, 9.0, 11.0, 13.0]:
    #         parameters = PARAMETERS.add(USE_ZB2020='True',SMAG_BI_CONST=0.06,ZB_SCALING=ZB_SCALING,STRESS_SMOOTH_PASS=1,STRESS_SMOOTH_SEL=2,VG_SHARP_PASS=2,VG_SHARP_SEL=2,VG_SMOOTH_PASS=1,VG_SMOOTH_SEL=4).add(**configuration(conf))
    #         ntasks = dict(R2=1, R3=1, R4=1, R5=1, R6=10, R7=10, R8=10)[conf]
    #         hpc = HPC.add(mem=4, ntasks=ntasks)
    #         run_experiment(f'/scratch/pp2681/mom6/Apr2023/generalization/ZB-EXP205-{conf}/ZB-{ZB_SCALING}', hpc, parameters)

    # for conf in ['R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']:
    #     parameters = PARAMETERS.add(SMAG_BI_CONST=0.06).add(**configuration(conf))+JansenHeld
    #     ntasks = dict(R2=1, R3=1, R4=1, R5=1, R6=10, R7=10, R8=10)[conf]
    #     hpc = HPC.add(mem=4, ntasks=ntasks)
    #     run_experiment(f'/scratch/pp2681/mom6/Apr2023/generalization/Jansen-Held-{conf}/ref', hpc, parameters)