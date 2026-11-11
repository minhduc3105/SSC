#!/usr/bin/env bash
# Export a fresh, source-only repository. Never changes the original Git history.
set -euo pipefail

usage() {
  echo "Usage: bash r2s100k/publish_code.sh GITHUB_URL [--push]"
  echo "Default: create a local clean repository only. --push also pushes main."
  echo "Optional: EXPORT_DIR=/absolute/new/directory (must not exist)."
}
if [[ $# -lt 1 || $# -gt 2 ]]; then usage; exit 2; fi
remote="$1"
push=false
if [[ $# -eq 2 ]]; then
  [[ "$2" == --push ]] || { usage; exit 2; }
  push=true
fi
case "$remote" in
  https://github.com/*/*|git@github.com:*/*) ;;
  *) echo 'Use https://github.com/OWNER/REPO.git or git@github.com:OWNER/REPO.git' >&2; exit 2 ;;
esac
for tool in git rsync python3; do command -v "$tool" >/dev/null; done
source_root="$(cd "$(dirname "$0")/.." && pwd)"
destination="${EXPORT_DIR:-$(dirname "$source_root")/SSC-code-$(date +%Y%m%d-%H%M%S)}"
destination="$(realpath -m "$destination")"
case "$destination/" in
  "$source_root/"*) echo 'Export must be outside the original SSC directory.' >&2; exit 2 ;;
esac
[[ ! -e "$destination" ]] || { echo "Destination already exists: $destination" >&2; exit 2; }
mkdir -p "$destination"

# Only source/configuration formats and specific original repository assets.
rsync -a --prune-empty-dirs \
  --exclude='.git/' --exclude='.venv*/' --exclude='venv/' \
  --exclude='.pip-cache*/' --exclude='.tmp*/' --exclude='__pycache__/' \
  --exclude='runs/' --exclude='checkpoints/' --exclude='wandb/' \
  --exclude='.env*' --exclude='node_modules/' \
  --include='/framework.png' --include='/voc12/cls_labels.npy' \
  --include='*/' --include='*.py' --include='*.sh' \
  --include='*.md' --include='*.txt' --include='*.json' \
  --include='*.yaml' --include='*.yml' --include='*.toml' \
  --include='*.cfg' --include='*.ini' --include='LICENSE*' \
  --include='COPYING*' --exclude='*' "$source_root/" "$destination/"

cat > "$destination/.gitignore" <<'EOF'
.venv*/
venv/
.pip-cache*/
.tmp*/
__pycache__/
*.py[cod]
.env
.env.*
r2s100k/runs/
checkpoints/
wandb/
data/
*.pth
*.pt
*.ckpt
*.npz
*.log
EOF

python3 - "$destination" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
files = [p for p in root.rglob('*') if p.is_file()]
large = [str(p.relative_to(root)) for p in files if p.stat().st_size > 20 * 1024**2]
if large:
    raise SystemExit(f'Unexpected files >20 MiB; export stopped: {large}')
print(f'Export: {len(files)} files, {sum(p.stat().st_size for p in files)/1024**2:.2f} MiB')
PY

git -C "$destination" init -b main
git -C "$destination" add .
if ! git -C "$destination" commit -m 'Add SSC source and R2S100K adapter setup'; then
  echo "Commit failed. Export remains at: $destination" >&2
  echo 'Configure git user.name/user.email, then commit and push from that directory.' >&2
  exit 1
fi
git -C "$destination" remote add origin "$remote"
echo "Clean repository: $destination"
if "$push"; then
  # No force push: an existing incompatible remote history is left untouched.
  git -C "$destination" push -u origin main
else
  echo "Review the export, then run:"
  printf 'git -C %q push -u origin main\n' "$destination"
fi
