#!/bin/bash
set -euo pipefail

SEEDS=$(seq 0 19)

OUTDIR="./output/emnist_sweep"
mkdir -p "$OUTDIR"

CLASS_COUNTS=($(python - <<'PY'
import numpy as np
vals = np.arange(0, 48).round().astype(int)
vals = np.unique(vals)
vals = vals[(vals >= 2) & (vals <= 47)]
if vals[-1] != 47:
    vals = np.append(vals, 47)
print(" ".join(map(str, vals)))
PY
))

for seed in $SEEDS; do
    for K in "${CLASS_COUNTS[@]}"; do
        LABEL="emnist_classes_${K}"
        echo "Running seed=${seed}, class count K=${K}"

        python ../../src/run_subliminal.py \
            --outdir "$OUTDIR" \
            --data-dir ../../EMNIST \
            --seed "$seed" \
            --sweep-name  "emnist_class_count_sweep" \
            --run-label "${LABEL}" \
            --dataset emnist \
            --emnist-split balanced \
            --class-count "${K}" \
            --class-selection first \
            --teacher-type mlp \
            --student-type mlp \
            --teacher-hidden-dims 256,256 \
            --student-hidden-dims 256,256 \
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
