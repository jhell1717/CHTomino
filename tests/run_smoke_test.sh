#!/usr/bin/env bash
# End-to-end pipeline smoke test: generates a tiny synthetic .zarr dataset,
# then runs compute_statistics.py -> train.py -> test.py against it with a
# small/fast model configuration. Verifies the whole pipeline runs without
# error and produces the expected artifacts; it does NOT check prediction
# accuracy (the synthetic data is fake).
#
# Usage: tests/run_smoke_test.sh
#
# Requires the project's Python environment to be active (see README.md) and
# `nvidia-dali` importable. On a machine without DALI installed (e.g. no
# NVIDIA GPU), point PYTHONPATH at devtools/dali_stub first -- see README.md
# "Environment notes" for why this is needed and why it's safe:
#   export PYTHONPATH="$(pwd)/devtools/dali_stub"

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

DATA_DIR="$WORK_DIR/data"
SCALING_FACTORS="$WORK_DIR/scaling_factors.pkl"
PROJECT_NAME="smoke_test"
OUTPUT_ROOT="$WORK_DIR/outputs"

echo "==> Generating synthetic .zarr dataset in $DATA_DIR"
python scripts/make_synthetic_zarr.py "$DATA_DIR" \
  --n-train 2 --n-val 1 --n-test 1 --n-tri 50 --n-surface-pts 200

# Small/fast overrides shared by every step below: tiny sampling counts (must
# stay <= the synthetic case sizes above), a tiny latent grid, CPU-only data
# handling, and an isolated output directory under $WORK_DIR so this never
# touches the repo's own outputs/.
COMMON_OVERRIDES=(
  "project.name=$PROJECT_NAME"
  "project.mlflow_db=$WORK_DIR/mlflow.db"
  "hydra.run.dir=$OUTPUT_ROOT/\${exp_tag}"
  "project_dir=$OUTPUT_ROOT/"
  "output=$OUTPUT_ROOT/\${exp_tag}"
  "resume_dir=$OUTPUT_ROOT/\${exp_tag}/models"
  "data.scaling_factors=$SCALING_FACTORS"
  "data.gpu_preprocessing=false"
  "data.gpu_output=false"
  "model.interp_res=[8,8,8]"
  "model.surface_points_sample=64"
  "model.geom_points_sample=64"
  "model.num_neighbors_surface=4"
)

echo "==> compute_statistics.py"
python compute_statistics.py \
  "${COMMON_OVERRIDES[@]}" \
  data.input_dir="$DATA_DIR/train" \
  data.max_samples_for_statistics=2

echo "==> train.py (2 epochs)"
python train.py \
  "${COMMON_OVERRIDES[@]}" \
  data.input_dir="$DATA_DIR/train" \
  data.input_dir_val="$DATA_DIR/val" \
  train.epochs=2 \
  train.checkpoint_interval=1 \
  train.amp.enabled=false

echo "==> test.py"
python test.py \
  "${COMMON_OVERRIDES[@]}" \
  eval.test_path="$DATA_DIR/test" \
  eval.save_path="$WORK_DIR/predicted"

echo "==> Checking expected artifacts exist"
test -f "$SCALING_FACTORS"
test -n "$(find "$OUTPUT_ROOT" -name 'DoMINO.*.pt' -print -quit)"
test -n "$(find "$WORK_DIR/predicted" -name '*_predicted.vtp' -print -quit)"
test -n "$(find "$WORK_DIR/predicted" -name '*_pred_vs_true.png' -print -quit)"

echo "==> Smoke test passed."
