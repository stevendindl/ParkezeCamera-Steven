#!/bin/bash
#SBATCH --job-name=park_yolo
#SBATCH --output=/work/%u/park-data/logs/%x_%j.out
#SBATCH --error=/work/%u/park-data/logs/%x_%j.err
#SBATCH --partition=gpu-v100-32gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sdindl@sc.edu

set -euo pipefail
echo "Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) started at $(date)"
echo "Running on: $SLURM_NODELIST"

# ----------------- Paths & variables -----------------
USER_WORK_DIR="/work/${USER}/park-data"
REMOTE_DATA_DIR="${USER_WORK_DIR}/data"
LOCAL_RUN_DIR="${TMPDIR:-/tmp}/${USER}_park_${SLURM_JOB_ID}"
RESULTS_REMOTE_DIR="${USER_WORK_DIR}/outputs/${SLURM_JOB_ID}"
LOG_DIR="${USER_WORK_DIR}/logs"
CODE_DIR="/home/${USER}/storage/park-code"   # where your train.py lives
# set this if you use a venv; otherwise set CONDA_ENV below
VENV_PATH="${CODE_DIR}/venv/bin/activate"
CONDA_ENV=""   # e.g. "myenv" if you use conda; leave empty if using venv

# make sure remote dirs exist
mkdir -p "${LOG_DIR}"
mkdir -p "${RESULTS_REMOTE_DIR}"

# safety checks
if [ ! -d "${REMOTE_DATA_DIR}" ]; then
  echo "ERROR: data not found at ${REMOTE_DATA_DIR}"
  exit 2
fi
if [ ! -d "${CODE_DIR}" ]; then
  echo "ERROR: code directory ${CODE_DIR} not found"
  exit 2
fi

echo "Copying data from ${REMOTE_DATA_DIR} -> ${LOCAL_RUN_DIR} (node-local)"
mkdir -p "${LOCAL_RUN_DIR}"
# use rsync for efficiency; this copies contents into local run dir
rsync -avh --progress --delete "${REMOTE_DATA_DIR}/" "${LOCAL_RUN_DIR}/"

# enter run dir
cd "${LOCAL_RUN_DIR}"
echo "Working dir: $(pwd)"
ls -lah

# ----------------- Load python / env -----------------
module load python3/anaconda/2019.10   # RCI recommended module

if [ -f "${VENV_PATH}" ]; then
  source "${VENV_PATH}"
  echo "Activated venv at ${VENV_PATH}"
else
  echo "No CONDA_ENV or VENV found; using system module python"
fi

# make tmp offload dir if model uses disk
mkdir -p "${LOCAL_RUN_DIR}/offload"

# Improve CUDA allocator behavior (helpful for fragmentation)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Optional NCCL environment tweaks to be polite on shared cluster (uncomment if needed)
# export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=^lo,docker

# ----------------- Run training -----------------
# Use absolute path for data yaml so training script doesn't need to assume cwd
DATA_YAML="${LOCAL_RUN_DIR}/data.yaml"
# If your data yaml is stored under a subdirectory in the data folder, update this:
# DATA_YAML="${LOCAL_RUN_DIR}/subdir/data.yaml"

if [ ! -f "${DATA_YAML}" ]; then
  # try the common alternative: /work path
  if [ -f "${REMOTE_DATA_DIR}/data.yaml" ]; then
    cp "${REMOTE_DATA_DIR}/data.yaml" "${LOCAL_RUN_DIR}/data.yaml"
  else
    echo "ERROR: data.yaml not found in ${REMOTE_DATA_DIR} or ${LOCAL_RUN_DIR}"
    ls -lah "${LOCAL_RUN_DIR}"
    exit 3
  fi
fi

# run your train script (adjust flags as needed)
echo "Starting training at $(date)"
# adjust this python command to your script's interface if different
python -u "${CODE_DIR}/train.py" \
  --config "${DATA_YAML}" \
  --output "${LOCAL_RUN_DIR}/results" \
  || { echo "Training failed"; exit 4; }

# ----------------- Sync results back -----------------
echo "Syncing results to ${RESULTS_REMOTE_DIR}"
mkdir -p "${RESULTS_REMOTE_DIR}"
rsync -avh --progress "${LOCAL_RUN_DIR}/results/" "${RESULTS_REMOTE_DIR}/"
# copy whole run dir except large venvs if any
rsync -avh --progress "${LOCAL_RUN_DIR}/" "${RESULTS_REMOTE_DIR}/" --exclude 'venv/' --exclude '__pycache__/' --exclude '*.tmp'

# optional cleanup of local run directory
echo "Cleaning node-local run dir ${LOCAL_RUN_DIR}"
rm -rf "${LOCAL_RUN_DIR}"

echo "Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) finished at $(date)"

