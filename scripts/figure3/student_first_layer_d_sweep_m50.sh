#!/bin/bash
set -euo pipefail

STUDENT_D1S=(8 11 16 23 32 45 64 91 128 181 256 362 512 724 1024 1448 2048 2896 4096)
SEEDS=$(seq 0 19)

OUTDIR="./output/student_first_layer_d_sweep_m50"
mkdir -p "$OUTDIR"

for seed in $SEEDS; do
    for D1 in "${STUDENT_D1S[@]}"; do
        LABEL="student_fc1_${D1}"

        echo "Running seed=${seed}, student first-layer D1=${D1}"

        python ../../src/run_subliminal.py \
            --outdir "$OUTDIR" \
            --data-dir ../../MNIST \
            --seed "$seed" \
            --sweep-name "mlp_baseline_sweep_student_first_layer_${D1}" \
            --run-label "$LABEL" \
            --dataset mnist \
            --teacher-type mlp \
            --student-type mlp \
            --teacher-hidden-dims 256,256 \
            --student-hidden-dims "${D1}",256 \
            --m 50 \
            --teacher-init A,A,A,A \
            --student-init random,random,A,A \
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