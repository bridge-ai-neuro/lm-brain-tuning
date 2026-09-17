#!/usr/bin/env bash
#SBATCH --job-name=lm-brain-tuning
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: sbatch --array=1-N $0 COMMAND_FILE" >&2
  exit 2
fi

command=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")
test -n "$command"
bash -lc "$command"
