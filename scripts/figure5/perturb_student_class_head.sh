#!/bin/bash
set -euo pipefail
SEEDS=$(seq 0 19)
OUTDIR="./outputs/student_class_head_perturb"
mkdir -p "$OUTDIR"
PERTURB_STDS=($(python - <<'PY'
import numpy as np
vals = np.linspace(0.0, 0.2, 40)
print(" ".join(f"{v:.8f}" for v in vals))
PY
))
for seed in $SEEDS; do
  for PSTD in "${PERTURB_STDS[@]}"; do
    LABEL=$(python - <<PY
p = float("${PSTD}")
print(f"student_class_head_perturb_{p:.8f}".replace(".", "p"))
PY
)
    echo "Running seed=${seed}, perturb_std=${PSTD}"
    python ../../src/run_subliminal.py \
      --outdir "$OUTDIR" \
      --data-dir ../../MNIST \
      --seed "$seed" \
      --sweep-name "mlp_student_class_head_perturb_sweep" \
      --run-label "$LABEL" \
      --dataset mnist \
      --teacher-type mlp \
      --student-type mlp \
      --teacher-hidden-dims 256,256 \
      --student-hidden-dims 256,256 \
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
      --noise-dist uniform \
      --perturb "student:class_head,std=${PSTD},timing=before_student_training,include_weight=true,include_bias=true"
  done
done