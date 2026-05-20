#!/bin/bash
set -euo pipefail
D_VALUES=(1 2 3 4 5 6 7 8 9 10 12 16 23 32 45 64 91 128 181 256 362 512 724 1024 1448 2048 2896 4096 5793 8192 11585)
SEEDS=$(seq 0 19)
OUTDIR="./outputs/d_sweep_baseline_emnist"
mkdir -p "$OUTDIR"
for seed in $SEEDS; do
  for D in "${D_VALUES[@]}"; do
    echo "Running seed=${seed}, D=${D}"
    python ../../src/run_subliminal.py \
      --outdir "$OUTDIR" \
      --data-dir ../../EMNIST \
      --seed "$seed" \
      --sweep-name "mlp_baseline_sweep_m" \
      --run-label "d${D}_seed${seed}" \
      --dataset emnist \
      --teacher-type mlp \
      --student-type mlp \
      --teacher-hidden-dims 256,"${D}" \
      --student-hidden-dims 256,"${D}" \
      --m 10 \
      --teacher-init A,A,A,A \
      --student-init A,A,A,A \
      --teacher-trainable true,true,true,true \
      --student-trainable true,true,true,true \
      --teacher-epochs 5 \
      --student-epochs 5 \
      --data-bsize 1024 \
      --noise-bsize 1000 \
      --noise-steps 60 \
      --noise-dist uniform
  done
done