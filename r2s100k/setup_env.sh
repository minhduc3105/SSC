#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PIP_CACHE_DIR="$PWD/.pip-cache-r2s100k"
export TMPDIR="$PWD/.tmp-r2s100k"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
if ! command -v python3.10 >/dev/null 2>&1; then
  echo 'Python 3.10 is required to create the SSC R2S100K environment.' >&2
  exit 2
fi
python3.10 -m venv .venv-r2s100k
.venv-r2s100k/bin/python -m pip install --upgrade pip setuptools wheel
.venv-r2s100k/bin/python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
.venv-r2s100k/bin/python -m pip install -r r2s100k/requirements.txt
.venv-r2s100k/bin/python -m pip install tensorboard seaborn pyyaml addict scikit-learn tqdm albumentations==1.4.24 opencv-python-headless==4.10.0.84
.venv-r2s100k/bin/python -m pip install --force-reinstall numpy==1.26.4
.venv-r2s100k/bin/python - <<'PY'
import torch, torchvision, skimage, pydensecrf
print('torch', torch.__version__, 'torchvision', torchvision.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available(): print(torch.cuda.get_device_name(0))
PY
