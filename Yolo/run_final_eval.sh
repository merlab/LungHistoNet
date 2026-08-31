#!/usr/bin/env bash
# wait for training to drain, then eval everything and export
set -u
cd /home/amir_ebrahimi/Documents/LungHistoNet/Yolo || exit 1
PY=/home/amir_ebrahimi/anaconda3/envs/tf/bin/python

while pgrep -f "train_merged.py" >/dev/null; do
  sleep 60
done

echo "=== training queue drained ==="
tr '\r' '\n' < runs/train_merged.log | grep -E "epochs completed|all training runs finished" | tail -4

echo "=== heatmap UNet ==="
"$PY" heatmap_counter.py eval 2>&1 | grep -vE "libtinfo|Warning|^[[:space:]]*$"

echo "=== full comparison ==="
"$PY" compare_models.py 2>&1 | grep -vE "libtinfo|Warning|^[[:space:]]*$"

echo "=== export winner ==="
"$PY" export_best.py --by f1 2>&1 | grep -vE "libtinfo|Warning|^[[:space:]]*$"
"$PY" export_best.py --by count_mae 2>&1 | grep -vE "libtinfo|Warning|^[[:space:]]*$"
echo "=== done ==="
