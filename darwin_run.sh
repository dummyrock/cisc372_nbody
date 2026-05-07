#!/usr/bin/env bash
set -euo pipefail

workgroup -g ea-cisc372-silber
vpkg_require gcc
vpkg_require cuda

echo "Host: $(hostname)"
echo "Date: $(date)"

make clean
make

# Run on one GPU-backed task.
salloc --ntasks=1 --cpus-per-task=1 --gpus=1 --partition=gpu-v100 srun ./nbody
