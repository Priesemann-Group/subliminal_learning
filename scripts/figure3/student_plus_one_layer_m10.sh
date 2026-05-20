#!/bin/bash
set -euo pipefail

SEEDS=$(seq 0 19)

OUTDIR="./output/student_plus_one_layer_m10"
mkdir -p "$OUTDIR"

for seed in $SEEDS; do
    echo "Running seed=${seed}"

    python ../../src/run_subliminal.py \
        --outdir "$OUTDIR" \
        --data-dir ../../MNIST \
        --seed "$seed" \
        --sweep-name "mlp_student_plus_one_layer" \
        --run-label "teacher_2hid_student_3hid_shared_heads" \
        --dataset mnist \
        --teacher-type mlp \
        --student-type mlp \
        --teacher-hidden-dims 256,256 \
        --student-hidden-dims 256,256,256 \
        --m 10 \
        --teacher-init fc1:random,fc2:random,class_head:A,aux_head:A \
        --student-init fc1:random,fc2:random,fc3:random,class_head:A,aux_head:A \
        --teacher-trainable all:true \
        --student-trainable all:true \
        --teacher-epochs 5 \
        --student-epochs 5 \
        --data-bsize 1024 \
        --noise-bsize 1000 \
        --noise-steps 60 \
        --noise-dist uniform
done
