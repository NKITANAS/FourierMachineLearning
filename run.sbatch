#!/bin/bash
#SBATCH -A TRN046
#SBATCH -J fourier_series_ai
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 1:00:00
#SBATCH -p batch
#SBATCH -N 1

set -euo pipefail

echo "Job start: $(date)"

# Load required modules if the `module` command is available
if command -v module >/dev/null 2>&1; then
	module load PrgEnv-gnu/8.7.0
	module load cpe/26.03
	module load miniforge3/23.11.0-0
	module load rocm/7.1.1
	module load craype-accel-amd-gfx90a
fi

# Limit threading for BLAS/OpenMP libraries
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

# Because using a non-default CPE
if [ -n "${CRAY_LD_LIBRARY_PATH:-}" ]; then
	export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:$LD_LIBRARY_PATH"
fi

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

# Allow overriding which script to run: pass as first arg or set SCRIPT env var.
SCRIPT=${1:-${SCRIPT:-train_model.py}}

echo "Running with Python: $(python -V 2>&1)"
echo "Executing script: $SCRIPT"

# Run Python script (unbuffered)
python -u "$SCRIPT"

echo "Job end: $(date)"