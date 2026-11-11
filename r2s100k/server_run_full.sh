#!/usr/bin/env bash
set -Eeuo pipefail

SSC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SSC_ROOT"

DATA_ROOT="${DATA_ROOT:-/srv/data/r2s100k}"
RUN_DIR="${RUN_DIR:-$SSC_ROOT/r2s100k/runs/full}"
CAM_BATCH_SIZE="${CAM_BATCH_SIZE:-6}"
IRN_BATCH_SIZE="${IRN_BATCH_SIZE:-16}"
CAM_EPOCHS="${CAM_EPOCHS:-10}"
IRN_EPOCHS="${IRN_EPOCHS:-3}"
CAM_LR="${CAM_LR:-0.1}"
IRN_LR="${IRN_LR:-0.1}"
MIN_FREE_GB="${MIN_FREE_GB:-100}"
PY="$SSC_ROOT/.venv-r2s100k/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "Missing $PY; run bash r2s100k/setup_env.sh first." >&2
  exit 2
fi
if [[ ! -d "$DATA_ROOT/train" || ! -d "$DATA_ROOT/train_labels/Train-Labels" ]]; then
  echo "Dataset must contain train/ and train_labels/Train-Labels/: $DATA_ROOT" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

available_gb="$(df -BG --output=avail "$RUN_DIR" | tail -n 1 | tr -dc '0-9')"
if (( available_gb < MIN_FREE_GB )); then
  echo "Only ${available_gb} GB free at $RUN_DIR; need at least ${MIN_FREE_GB} GB for manifests and generated artifacts." >&2
  exit 2
fi

echo "SSC_ROOT=$SSC_ROOT"
echo "DATA_ROOT=$DATA_ROOT"
echo "RUN_DIR=$RUN_DIR"
echo "Free disk=${available_gb}GB; required minimum=${MIN_FREE_GB}GB"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -c 'import torch; print("torch",torch.__version__,"CUDA",torch.cuda.is_available()); assert torch.cuda.is_available(); print("GPU",torch.cuda.get_device_name(0))'

run_stage() {
  local name="$1"
  shift
  echo "[$(date --iso-8601=seconds)] START $name"
  "$@" 2>&1 | tee "$LOG_DIR/$name.log" | tee -a "$LOG_DIR/pipeline.log"
  echo "[$(date --iso-8601=seconds)] DONE $name"
}

run_stage prepare "$PY" r2s100k/pipeline.py prepare \
  --dataset-root "$DATA_ROOT" --output "$RUN_DIR"
run_stage superpixels "$PY" r2s100k/pipeline.py superpixels \
  --manifest "$RUN_DIR/train_images.json" --output "$RUN_DIR/superpixels"
run_stage train-cam "$PY" r2s100k/pipeline.py train-cam \
  --manifest "$RUN_DIR/train_images.json" --superpixels "$RUN_DIR/superpixels" \
  --output "$RUN_DIR/cam_checkpoint.pt" --device cuda --workers 4 \
  --batch-size "$CAM_BATCH_SIZE" --epochs "$CAM_EPOCHS" --lr "$CAM_LR" --resume
run_stage make-cam "$PY" r2s100k/pipeline.py make-cam \
  --manifest "$RUN_DIR/train_images.json" --checkpoint "$RUN_DIR/cam_checkpoint.pt" \
  --output "$RUN_DIR/cams" --device cuda --long-side 512 --scales 1 0.5 1.5 2
run_stage make-cam-masks "$PY" r2s100k/pipeline.py make-cam-masks \
  --manifest "$RUN_DIR/train_images.json" --cams "$RUN_DIR/cams" \
  --output "$RUN_DIR/cam_masks" --bg-threshold 0.25
run_stage evaluate-cam "$PY" r2s100k/pipeline.py evaluate \
  --manifest "$RUN_DIR/train_images.json" --labels "$RUN_DIR/train_labels.json" \
  --predictions "$RUN_DIR/cam_masks" --output "$RUN_DIR/evaluation_train_cam"
run_stage make-ir-labels "$PY" r2s100k/pipeline.py make-ir-labels \
  --manifest "$RUN_DIR/train_images.json" --cams "$RUN_DIR/cams" \
  --output "$RUN_DIR/ir_labels" --fg-threshold 0.30 --bg-threshold 0.10
run_stage train-irn "$PY" r2s100k/pipeline.py train-irn \
  --manifest "$RUN_DIR/train_images.json" --labels "$RUN_DIR/ir_labels" \
  --output "$RUN_DIR/irn_checkpoint.pt" --device cuda --workers 4 \
  --batch-size "$IRN_BATCH_SIZE" --epochs "$IRN_EPOCHS" --lr "$IRN_LR" \
  --weight-decay 1e-4 --resume
run_stage make-pseudo "$PY" r2s100k/pipeline.py make-pseudo \
  --manifest "$RUN_DIR/train_images.json" --cams "$RUN_DIR/cams" \
  --checkpoint "$RUN_DIR/irn_checkpoint.pt" --output "$RUN_DIR/pseudo_masks" \
  --scores "$RUN_DIR/scores" --device cuda --long-side 512 \
  --beta 10 --exp-times 8 --bg-threshold 0.25
run_stage evaluate-pseudo "$PY" r2s100k/pipeline.py evaluate \
  --manifest "$RUN_DIR/train_images.json" --labels "$RUN_DIR/train_labels.json" \
  --predictions "$RUN_DIR/pseudo_masks" --output "$RUN_DIR/evaluation_train_pseudo"
run_stage make-cam-test "$PY" r2s100k/pipeline.py make-cam \
  --manifest "$RUN_DIR/test_images.json" --checkpoint "$RUN_DIR/cam_checkpoint.pt" \
  --output "$RUN_DIR/test_cams" --device cuda --long-side 512 \
  --scales 1 0.5 1.5 2 --class-mode predicted --class-threshold 0.0
run_stage make-cam-masks-test "$PY" r2s100k/pipeline.py make-cam-masks \
  --manifest "$RUN_DIR/test_images.json" --cams "$RUN_DIR/test_cams" \
  --output "$RUN_DIR/test_cam_masks" --bg-threshold 0.25
run_stage make-pseudo-test "$PY" r2s100k/pipeline.py make-pseudo \
  --manifest "$RUN_DIR/test_images.json" --cams "$RUN_DIR/test_cams" \
  --checkpoint "$RUN_DIR/irn_checkpoint.pt" --output "$RUN_DIR/test_pseudo_masks" \
  --scores "$RUN_DIR/test_scores" --device cuda --long-side 512 \
  --beta 10 --exp-times 8 --bg-threshold 0.25
run_stage evaluate-cam-test "$PY" r2s100k/pipeline.py evaluate \
  --manifest "$RUN_DIR/test_images.json" --labels "$RUN_DIR/test_labels.json" \
  --predictions "$RUN_DIR/test_cam_masks" --output "$RUN_DIR/evaluation_test_cam"
run_stage evaluate-pseudo-test "$PY" r2s100k/pipeline.py evaluate \
  --manifest "$RUN_DIR/test_images.json" --labels "$RUN_DIR/test_labels.json" \
  --predictions "$RUN_DIR/test_pseudo_masks" --output "$RUN_DIR/evaluation_test_pseudo"

run_stage compare-metrics "$PY" - "$RUN_DIR" <<'PY'
import json, pathlib, sys
run = pathlib.Path(sys.argv[1])
for split in ('train', 'test'):
    cam = json.loads((run/f'evaluation_{split}_cam'/'metrics.json').read_text())
    pseudo = json.loads((run/f'evaluation_{split}_pseudo'/'metrics.json').read_text())
    print(f'[{split}]')
    for key in ('miou_15', 'miou_foreground14', 'foreground_binary_iou', 'foreground_binary_dice'):
        a, b = cam.get(key), pseudo.get(key)
        delta = None if a is None or b is None else b-a
        print(f'{key}: CAM={a} pseudo={b} delta={delta}')
PY

echo "Full run complete. Per-stage logs: $LOG_DIR/; metrics: evaluation_{train,test}_{cam,pseudo}"
