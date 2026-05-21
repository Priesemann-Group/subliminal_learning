#!/bin/bash
set -euo pipefail

SEEDS=$(seq 0 19)

OUTDIR="./output/mlp_teach_cnn_stud_m50"
mkdir -p "$OUTDIR"


STUDENT_CNN_SPEC="${OUTDIR}/student_cnn_32_128_pool4.json"

cat > "${STUDENT_CNN_SPEC}" <<'JSON'
[
  {
    "name": "conv1",
    "type": "conv2d",
    "out_channels": 32,
    "kernel_size": 3,
    "padding": 1,
    "activation": "relu",
    "pool": {"type": "max", "kernel_size": 2}
  },
  {
    "name": "conv2",
    "type": "conv2d",
    "out_channels": 128,
    "kernel_size": 3,
    "padding": 1,
    "activation": "relu",
    "pool": {"type": "max", "kernel_size": 4}
  },
  {
    "name": "fc1",
    "type": "linear",
    "out_features": 256,
    "activation": "relu"
  }
]
JSON

for seed in $SEEDS; do
    echo "Running seed=${seed}"

    python ../../src/run_subliminal.py \
    --outdir "${OUTDIR}" \
    --data-dir ../../MNIST \
    --seed "$seed" \
    --sweep-name "mlp_teacher_cnn_student_matched_params" \
    --run-label "teacher_mlp_3x256_student_cnn_32_128_fc256_m50" \
    --dataset mnist \
    --teacher-type mlp \
    --student-type cnn \
    --teacher-hidden-dims 256,256,256 \
    --student-hidden-dims 256 \
    --student-arch-spec "${STUDENT_CNN_SPEC}" \
    --m 50 \
    --teacher-init fc1:random,fc2:random,fc3:random,class_head:A,aux_head:A \
    --student-init conv1:random,conv2:random,fc1:random,class_head:A,aux_head:A \
    --teacher-trainable all:true \
    --student-trainable all:true \
    --teacher-epochs 5 \
    --student-epochs 5 \
    --data-bsize 1024 \
    --noise-bsize 1000 \
    --noise-steps 60 \
    --noise-dist perlin \
    --perlin-res 8
done


