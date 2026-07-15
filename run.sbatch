#!/bin/bash
#SBATCH -A TRN046
#SBATCH -J fourier_series_ai
#SBATCH -o %x-%j.out
#SBATCH -t 1:00:00
#SBATCH -p batch
#SBATCH -N 1

date

module load PrgEnv-gnu/8.7.0
module load cpe/26.03
module load miniforge3/23.11.0-0
module load rocm/7.1.1
module load craype-accel-amd-gfx90a

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Because using a non-default CPE
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH

export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

source .venv/bin/activate

python -u cnn.py