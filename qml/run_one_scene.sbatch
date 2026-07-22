#!/bin/bash
#SBATCH -A TRN046
#SBATCH -J fourier_series_qml_one_scene
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 10:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7

set -euo pipefail
echo "Job start: $(date)"

# Load required modules
if command -v module >/dev/null 2>&1; then
	module load PrgEnv-gnu/8.7.0
	module load cpe/26.03
	module load miniforge3/23.11.0-0
	module load rocm/7.1.1
	module load craype-accel-amd-gfx90a
fi

# Threading for BLAS/OpenMP libraries — match cpus-per-task above rather
# than forcing single-threaded, since data generation is CPU-bound NumPy work.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-$SLURM_CPUS_PER_TASK}

# Because using a non-default CPE
if [ -n "${CRAY_LD_LIBRARY_PATH:-}" ]; then
	export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:$LD_LIBRARY_PATH"
fi

# Make sure the allocated GPU is actually visible to ROCm/PyTorch
export ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-$SLURM_LOCALID}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-$SLURM_LOCALID}

# Create a MIOPEN cache directory in /tmp (unique per job)
MIOPEN_USER_DB_PATH=${MIOPEN_USER_DB_PATH:-"/tmp/my-miopen-cache-$SLURM_JOB_ID"}
export MIOPEN_USER_DB_PATH
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf "${MIOPEN_USER_DB_PATH}" || true
mkdir -p "${MIOPEN_USER_DB_PATH}"
trap 'rm -rf "${MIOPEN_USER_DB_PATH}" || true' EXIT

# Project directory (defaults to current working directory when submitting)
PROJECT_DIR=${PROJECT_DIR:-$PWD}
cd "$PROJECT_DIR"

# Activate virtualenv if present; otherwise fall back to system/conda python
VENV_ACTIVATE="$PROJECT_DIR/.venv/bin/activate"
if [ -f "$VENV_ACTIVATE" ]; then
	# shellcheck disable=SC1091
	source "$VENV_ACTIVATE"
	echo "Activated virtualenv: $VENV_ACTIVATE"
else
	echo "Warning: virtualenv not found at $VENV_ACTIVATE. Using system Python."
fi

# Requires the `pennylane` package in addition to the classical stack
# (e.g. `pip install pennylane`) for the quantum circuit used by qml/train_one_scene_model.py.

# Allow overriding which script to run: pass as first arg or set SCRIPT env var.
SCRIPT=${1:-${SCRIPT:-qml/train_one_scene_model.py}}
echo "Running with Python: $(python -V 2>&1)"
echo "Executing script: $SCRIPT"
echo "GPUs visible to job: $(rocm-smi --showid 2>/dev/null || echo 'rocm-smi not available')"

# Run through srun so the task is properly bound to its allocated GPU/CPUs
srun --gpus-per-task=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python -u "$SCRIPT"

echo "Job end: $(date)"
