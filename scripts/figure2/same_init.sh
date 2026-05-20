#!/bin/bash
set -euo pipefail

m_VALUES=(3 10 25 50 100 250)
SEEDS=$(seq 0 1)

RESULT_DIR="./output/same_init"
mkdir -p "$RESULT_DIR"

for seed in $SEEDS; do
    for m in "${m_VALUES[@]}"; do
        echo "Running seed=${seed}, m=${m}"

        python ../../src/run_subliminal.py \
            --dataset mnist \
            --data-dir /data/nst/vbrockers/projects/subliminal_learning_private/MNIST_DATA \
            --outdir "$RESULT_DIR" \
            --seed "$seed" \
            --teacher-epochs 5 \
            --student-epochs 5 \
            --data-bsize 1024 \
            --noise-bsize 1000 \
            --noise-steps 60 \
            --m "$m" \
            --init-config-teacher "A,A,A,A" \
            --trainable-config-teacher "true,true,true,true" \
            --init-config-student "A,A,A,A" \
            --trainable-config-student "true,true,true,true" \
            --noise-dist uniform
    done
done