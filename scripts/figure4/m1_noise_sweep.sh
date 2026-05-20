#!/bin/bash
set -euo pipefail
OUTDIR="./outputs/m1_noise_sweep"
mkdir -p "$OUTDIR"
SEEDS=($(seq 0 19))
MS=(1)

NOISE_STEPS_LIST=($(python - <<'PY'
import numpy as np
vals = (np.sqrt(2) ** np.arange(1, 27)).round().astype(int)
vals = np.unique(vals)
print(" ".join(map(str, vals)))
PY
))

echo "Seeds: ${SEEDS[*]}"
echo "m values: ${MS[*]}"
echo "noise steps values: ${NOISE_STEPS_LIST[*]}"

for SEED in "${SEEDS[@]}"; do
  for M in "${MS[@]}"; do
    for NSTEPS in "${NOISE_STEPS_LIST[@]}"; do
      LABEL="seed${SEED}_m${M}_noise_steps_${NSTEPS}"
      python ../../src/run_subliminal.py \
        --outdir "${OUTDIR}" \
        --data-dir ../../MNIST \
        --seed "${SEED}" \
        --sweep-name "mlp_m_noise_steps_grid" \
        --run-label "${LABEL}" \
        --dataset mnist \
        --teacher-type mlp \
        --student-type mlp \
        --teacher-hidden-dims 256,256 \
        --student-hidden-dims 256,256 \
        --m "${M}" \
        --teacher-init A,A,A,A \
        --student-init A,A,A,A \
        --teacher-trainable true,true,true,true \
        --student-trainable true,true,true,true \
        --teacher-epochs 5 \
        --student-epochs 5 \
        --data-bsize 1024 \
        --noise-bsize 1000 \
        --noise-steps "${NSTEPS}" \
        --noise-dist uniform
    done
  done
done