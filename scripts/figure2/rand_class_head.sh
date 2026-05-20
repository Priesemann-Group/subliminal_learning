#!/bin/bash
set -euo pipefail

m_VALUES=(3 10 25 50 100 250)
SEEDS=$(seq 0 19)

OUTDIR="./output/random_class_head"
mkdir -p "$OUTDIR"

for seed in $SEEDS; do
    for m in "${m_VALUES[@]}"; do
        echo "Running seed=${seed}, m=${m}"

        python ../../src/run_subliminal.py \
            --dataset mnist \
            --data-dir ../../MNIST \
            --outdir "$OUTDIR" \
            --seed "$seed" \
            --teacher-epochs 5 \
            --student-epochs 5 \
            --data-bsize 1024 \
            --noise-bsize 1000 \
            --noise-steps 60 \
            --m "$m" \
            --teacher-init "A,A,A,A" \
            --student-init "A,A,A,random" \
            --teacher-trainable "true,true,true,true" \
            --student-trainable "true,true,true,true" \
            --noise-dist uniform
    done
done