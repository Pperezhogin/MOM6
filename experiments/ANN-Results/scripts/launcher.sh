#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --begin=now
#SBATCH --time=48:00:00
#SBATCH --job-name=ANN_training

### For training: ntasks=4, mem=64GB, time=48:00:00

echo " "
scontrol show jobid -dd $SLURM_JOB_ID
echo " "
echo "The number of alphafold processes:"
ps -e | grep -i alphafold | wc -l
echo " "
module purge

#singularity exec --nv --overlay /scratch/pp2681/python-container/python-overlay.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "source /ext3/env.sh; time python -u generate_datasets.py --operator_str=\"Filtering(FGR=2,shape=gcm_filters.FilterShape.TAPER)+CoarsenKochkov()\" "
#singularity exec --nv --overlay /scratch/pp2681/python-container/python-overlay.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "source /ext3/env.sh; time python -u train_script.py --exp=hdn-64-64-sym-trev"
singularity exec --nv --overlay /scratch/pp2681/python-container/python-overlay.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "source /ext3/env.sh; time python -u generate_3d_datasets.py --factor=15 "
#singularity exec --nv --overlay /scratch/pp2681/python-container/python-overlay.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "source /ext3/env.sh; time python -u train_script.py --feature_functions=\"[]\" --stencil_size=5 --path_save=features_stencil/stencil-5-no-radius "
