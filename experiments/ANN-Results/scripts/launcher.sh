#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64GB
#SBATCH --begin=now
#SBATCH --time=48:00:00
#SBATCH --job-name=CM26_ds

#SBATCH --output=FGR2_Taper.out
#SBATCH --error=FGR2_Taper.err

echo " "
scontrol show jobid -dd $SLURM_JOB_ID
echo " "
echo "The number of alphafold processes:"
ps -e | grep -i alphafold | wc -l
echo " "
module purge

singularity exec --nv --overlay /scratch/pp2681/python-container/python-overlay.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "source /ext3/env.sh; time python -u generate_datasets.py --operator_str=\"Filtering(FGR=2,shape=gcm_filters.FilterShape.TAPER)+CoarsenKochkov()\" "