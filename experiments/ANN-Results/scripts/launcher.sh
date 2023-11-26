#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=32GB
#SBATCH --begin=now
#SBATCH --time=24:00:00
#SBATCH --job-name=CM26_ANN

#SBATCH --output=hidden-64-64-symmetries-revers.out
#SBATCH --error=hidden-64-64-symmetries-revers.err

echo " "
scontrol show jobid -dd $SLURM_JOB_ID
echo " "
echo "The number of alphafold processes:"
ps -e | grep -i alphafold | wc -l
echo " "
module purge

#singularity exec --nv --overlay /scratch/pp2681/python-container/python-overlay.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "source /ext3/env.sh; time python -u generate_datasets.py --operator_str=\"Filtering(FGR=2,shape=gcm_filters.FilterShape.TAPER)+CoarsenKochkov()\" "
singularity exec --nv --overlay /scratch/pp2681/python-container/python-overlay.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "source /ext3/env.sh; time python -u train_script.py --path_save=/scratch/pp2681/mom6/CM26_ML_models/Gauss-FGR2/hidden-64-64-symmetries-revers --time_revers=\"True\" --hidden_layers=\"[64,64]\" --symmetries=\"True\""