#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28GB
#SBATCH --begin=now
#SBATCH --time=06:00:00
#SBATCH --job-name=ANN_training

### For training: ntasks=4, mem=64GB, time=48:00:00
### For filtering: --cpus-per-task=14 --mem=64GB time=48:00:00
### For training fluxes:  --cpus-per-task=14 --mem=30GB time=06:00:00 (estimated training time is 2.4 hours + 20 mins on testing)
### For training fluxes:  --cpus-per-task=8 --mem=28GB time=06:00:00 (estimated training time is 3.5 hours + 30 mins on testing) -- this is preferable because on one node 6 ANNs can be trained simultaneously

echo " "
scontrol show jobid -dd $SLURM_JOB_ID
echo " "
echo "The number of alphafold processes:"
ps -e | grep -i alphafold | wc -l
echo " "
module purge

#singularity exec --nv --overlay /scratch/pp2681/python-container/python-overlay.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "source /ext3/env.sh; time python -u generate_datasets.py --operator_str=\"Filtering(FGR=2,shape=gcm_filters.FilterShape.TAPER)+CoarsenKochkov()\" "
#singularity exec --nv --overlay /scratch/pp2681/python-container/python-overlay.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "source /ext3/env.sh; time python -u train_script.py --exp=hdn-64-64-sym-trev"
#singularity exec --nv --overlay /scratch/pp2681/python-container/python-overlay.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "source /ext3/env.sh; time python -u generate_3d_datasets.py --factor=9 "
#singularity exec --nv --overlay /scratch/pp2681/python-container/python-overlay.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "source /ext3/env.sh; time python -u train_script.py --stencil_size=1 --gradient_features=\"['sh_xy', 'sh_xx', 'rel_vort']\" --path_save=strain-models/1x1 "
#singularity exec --nv --overlay /scratch/pp2681/python-container/python-overlay.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "source /ext3/env.sh; time python -u train_script.py --feature_functions=\"[deformation_radius_over_grid_spacing_linear, Held_Larichev_1996, square_root_of_Ri, rescaled_depth]\" --hidden_layers=\"[32,32]\" --stencil_size=3  --path_save=3x3-32-32-all-nd-numbers-test "

singularity exec --nv --overlay /scratch/pp2681/python-container/python-overlay.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "source /ext3/env.sh; time python -u train_script_fluxes.py --hidden_layers=\"[20]\" --path_save=flux-models/20-seed0 "
