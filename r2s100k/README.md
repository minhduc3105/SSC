# SSC adapter for R2S100K

All adapter code is isolated here; the original SSC files are not patched. R2S100K labels are background `0` and foreground `1..14`. Image-level labels are read from train GT; training loaders only consume those vectors, images, and SLIC maps.

## Data preparation

```bash
cd SSC
.venv-r2s100k/bin/python r2s100k/pipeline.py prepare \
  --dataset-root /run/media/minhduc/D/Study/r2s100k \
  --output r2s100k/runs/r2s100k
```

Preparation now prints the split counts, the start/end of the held-out hash audit, and progress every 250 files. Other stages print their configuration, periodic sample or batch progress, elapsed time/ETA, and a completion or failure line. The first model run downloads the ImageNet ResNet-50 and VGG19 weights if they are not already cached.

## Local end-to-end smoke test

Use this on the local GPU to check that data, CAM training/inference, CRF affinity labels, IRNet training, random walk, mask writing, and evaluation are connected. It uses four deterministic train images, batch size 1, three CAM epochs (so the SSC alignment loss is exercised after warm-up), one IRNet epoch, and 256-pixel inference. It then predicts four test images using classifier-inferred classes, with test GT only passed to evaluation. These are smoke settings only; do not treat the metrics as quality results or use them for the 3090 full-run configuration.

```bash
.venv-r2s100k/bin/python r2s100k/pipeline.py smoke \
  --prepared r2s100k/runs/data \
  --output r2s100k/runs/smoke-local \
  --count 4 --device cuda
```

If a late stage fails after its earlier artifacts were created, fix the code and rerun with `--reuse-existing`; the command verifies expected files exist before skipping a stage. To rebuild every smoke artifact, use a new output directory or remove the old smoke directory.

## Superpixels and CAM

Install `r2s100k/requirements.txt` in the SSC environment, then:

```bash
.venv-r2s100k/bin/python r2s100k/pipeline.py superpixels \
  --manifest r2s100k/runs/r2s100k/train_images.json \
  --output r2s100k/runs/r2s100k/superpixels

.venv-r2s100k/bin/python r2s100k/pipeline.py train-cam \
  --manifest r2s100k/runs/r2s100k/train_images.json \
  --superpixels r2s100k/runs/r2s100k/superpixels \
  --output r2s100k/runs/r2s100k/cam_checkpoint.pt \
  --device cuda --batch-size 6 --epochs 10
```

The CAM adapter uses 14 classifier outputs and an SSC reconstruction decoder whose first layer is adapted to 14 channels. It keeps SSC's ResNet-50 DRS backbone, multi-label classification objective, reconstruction/perceptual loss, and superpixel alignment objective.

## Remaining stages

Run these after CAM training. They produce compact CAM archives, a CAM-derived mask baseline, CRF affinity labels, the SSC IRNet checkpoint, and final random-walk pseudo-label PNGs:

```bash
.venv-r2s100k/bin/python r2s100k/pipeline.py make-cam --manifest r2s100k/runs/r2s100k/train_images.json --checkpoint r2s100k/runs/r2s100k/cam_checkpoint.pt --output r2s100k/runs/r2s100k/cams --device cuda
.venv-r2s100k/bin/python r2s100k/pipeline.py make-cam-masks --manifest r2s100k/runs/r2s100k/train_images.json --cams r2s100k/runs/r2s100k/cams --output r2s100k/runs/r2s100k/cam_masks --bg-threshold 0.25
.venv-r2s100k/bin/python r2s100k/pipeline.py evaluate --manifest r2s100k/runs/r2s100k/train_images.json --labels r2s100k/runs/r2s100k/train_labels.json --predictions r2s100k/runs/r2s100k/cam_masks --output r2s100k/runs/r2s100k/evaluation_cam
.venv-r2s100k/bin/python r2s100k/pipeline.py make-ir-labels --manifest r2s100k/runs/r2s100k/train_images.json --cams r2s100k/runs/r2s100k/cams --output r2s100k/runs/r2s100k/ir_labels
.venv-r2s100k/bin/python r2s100k/r2s100k/pipeline.py train-irn --manifest r2s100k/runs/r2s100k/train_images.json --labels r2s100k/runs/r2s100k/ir_labels --output r2s100k/runs/r2s100k/irn_checkpoint.pt --device cuda
.venv-r2s100k/bin/python r2s100k/pipeline.py make-pseudo --manifest r2s100k/runs/r2s100k/train_images.json --cams r2s100k/runs/r2s100k/cams --checkpoint r2s100k/runs/r2s100k/irn_checkpoint.pt --output r2s100k/runs/r2s100k/pseudo_masks --scores r2s100k/runs/r2s100k/scores --device cuda
.venv-r2s100k/bin/python r2s100k/pipeline.py evaluate --manifest r2s100k/runs/r2s100k/train_images.json --labels r2s100k/runs/r2s100k/train_labels.json --predictions r2s100k/runs/r2s100k/pseudo_masks --output r2s100k/runs/r2s100k/evaluation
```

## Server setup, data transfer, and full run

The dataset currently occupies about 9.5 GB. Generated SLIC maps are full-resolution `uint16`; CAM archives and auxiliary random-walk scores are stored compactly. Keep at least 100 GB free on the filesystem holding the run directory before starting the full run.

The server connection you provided is `root@38.246.237.140` on SSH port `30161`. The `-L 8080:localhost:8080` option forwards a remote web interface; it is optional for training. First connect and check which mounted path has enough persistent space for the run (at least 100 GB free):

```bash
ssh -p 30161 -L 8080:localhost:8080 root@38.246.237.140
df -h /root /tmp
lsblk
nvidia-smi
```

After choosing writable paths on the large disk, set them in a local terminal and transfer the adapter/repository and dataset. The examples below use `/root`; change `REMOTE_ROOT` if that filesystem does not have enough room:

```bash
cd /run/media/minhduc/D/project/KLTN
SERVER=root@38.246.237.140
SSH_PORT=30161
REMOTE_ROOT=/root
REMOTE_PROJECT="$REMOTE_ROOT/SSC"
REMOTE_DATA="$REMOTE_ROOT/data/r2s100k"

ssh -p "$SSH_PORT" "$SERVER" "mkdir -p '$REMOTE_PROJECT' '$REMOTE_DATA'"
rsync -avh --partial --info=progress2 \
  -e "ssh -p $SSH_PORT" \
  --exclude='.venv-r2s100k/' --exclude='.pip-cache-r2s100k/' \
  --exclude='.tmp-r2s100k/' --exclude='r2s100k/runs/' \
  --exclude='__pycache__/' \
  SSC/ "$SERVER:$REMOTE_PROJECT/"
rsync -avh --partial --info=progress2 \
  -e "ssh -p $SSH_PORT" \
  /run/media/minhduc/D/Study/r2s100k/ "$SERVER:$REMOTE_DATA/"
```

The trailing slash on the dataset source copies the contents into `REMOTE_DATA`. If transfer disconnects, rerun the same `rsync` commands. Then connect to the server and run the following there. These package commands assume Ubuntu/Debian; Python 3.10 is required by the setup script:

```bash
apt-get update
apt-get install -y python3.10 python3.10-venv python3.10-dev build-essential git tmux rsync
cd /root/SSC
bash r2s100k/setup_env.sh
.venv-r2s100k/bin/python -c 'import torch,numpy,pydensecrf; print(torch.__version__,numpy.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0))'
```

Keep the full run alive if SSH disconnects by using `tmux`. The script checks free disk and CUDA, runs every stage in sequence, supports CAM/IRNet checkpoint resume, and writes one log per stage plus a combined log:

```bash
cd /root/SSC
tmux new -s ssc-r2s100k
DATA_ROOT=/root/data/r2s100k RUN_DIR=/root/SSC/r2s100k/runs/full \
  bash r2s100k/server_run_full.sh
```

Detach from tmux with `Ctrl-b`, then `d`. To monitor progress or reconnect:

```bash
tail -f /root/SSC/r2s100k/runs/full/logs/pipeline.log
tmux attach -t ssc-r2s100k
```

Per-stage logs are in `runs/full/logs/`. Metrics are written to `evaluation_train_cam/`, `evaluation_train_pseudo/`, `evaluation_test_cam/`, and `evaluation_test_pseudo/`; the final `compare-metrics` stage prints CAM-to-pseudo deltas for both splits. On test, active CAM classes come from the trained image classifier at a fixed logit threshold of `0.0`; test GT is used only by the evaluator. Train labels are still derived from train GT, so train metrics are diagnostics, while test metrics are the independent evaluation (assuming the test split was not used for model or threshold selection).

## Full pipeline

The recommended server entry point is `r2s100k/server_run_full.sh`. Its defaults match the full configuration: CAM crop 512, batch 6, 10 epochs; IRNet crop 512, batch 16, 3 epochs; CAM scales `1, 0.5, 1.5, 2`; and random walk `beta=10`, `exp_times=8`. It creates and evaluates CAM and refined pseudo-masks on both train and test. Test classes are inferred from the CAM classifier; test GT is not passed into inference. Override batch, learning rate, or epoch counts with `CAM_BATCH_SIZE`, `CAM_LR`, `CAM_EPOCHS`, `IRN_BATCH_SIZE`, `IRN_LR`, or `IRN_EPOCHS` when needed. If reducing a batch to avoid OOM, scale that stage's LR by the same ratio (for example, halve CAM batch 6→3 and CAM LR 0.1→0.05). R2S100K inference resizes the long side to 512 before random-walk propagation, then writes semantic masks at original resolution.
