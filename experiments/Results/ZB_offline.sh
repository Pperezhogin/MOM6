#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5GB
#SBATCH --partition=cs
#SBATCH --job-name=ZB_offline
#SBATCH --time=1:0:00
#SBATCH --array=1024-2047
#SBATCH --output=/scratch/pp2681/mom6/offline/%a.out
#SBATCH --error=/scratch/pp2681/mom6/offline/%a.err

# Calculate the 6-dimensional indices from the 1D index
index=$SLURM_ARRAY_TASK_ID
i1=$(( (index / 1024) % 4 ))
i2=$(( (index / 256) % 4 ))
i3=$(( (index / 64) % 4 ))
i4=$(( (index / 16) % 4 ))
i5=$(( (index / 4) % 4 ))
i6=$(( index % 4 ))

# Print the 6-dimensional array indices
echo "i1=$i1 i2=$i2 i3=$i3 i4=$i4 i5=$i5 i6=$i6"

module purge
singularity exec --nv --overlay /scratch/pp2681/python-container/python-overlay.ext3:ro /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; time python -u ZB_offline.py --LPF_iter=$i1 --LPF_order=$i2 --HPF_iter=$i3 --HPF_order=$i4 --Stress_iter=$i5 --Stress_order=$i6 --file=\"/scratch/pp2681/mom6/offline/$SLURM_ARRAY_TASK_ID.nc\" "