#!/usr/bin/env bash
# One-command sync + remote-env management for the robustness repo, targeting cranberry.
#
# Source-of-truth split:
#   - CODE        lives on LOCAL (you edit here; `push` flows it UP, never touching data/).
#   - data/       lives ONLY on cranberry's scratch ($REMOTE_DATA_DIR). The Mac sees it via
#                 sshfs (`./sync.sh mount`), so no bulk data is stored locally.
#
# cranberry layout (everything bulk on the 7.3T scratch; $HOME is a 10G NFS quota -- code only):
#   $REMOTE_SCRATCH/robustness-data/{input,cache,output}   <- data/ (symlinked from ~/robustness/data)
#   $REMOTE_SCRATCH/miniforge3, /envs, /uv-cache, ...       <- conda+uv toolchain + caches
#
# Toolchain is standardized to conda + uv + ruff + pyproject.toml (no hand-managed venv):
#   conda env (python+uv+ruff, from environment.yml) -> `uv pip install -e .` -> ruff.
#
# Usage (run on LOCAL / the Mac):
#   ./sync.sh setup     one-time: create scratch data dirs + ownership README + data/ symlink on cranberry
#   ./sync.sh env       build/refresh the conda+uv environment on cranberry (idempotent)
#   ./sync.sh push      push code UP (local -> cranberry)
#   ./sync.sh input     push code UP and push data/input/ UP (run after staging a dataset locally)
#   ./sync.sh pull      pull data/output + data/cache DOWN to a local data/ (only if NOT using sshfs)
#   ./sync.sh mount     sshfs-mount cranberry's data/ onto local ./data  (needs macFUSE + sshfs-mac)
#   ./sync.sh umount    unmount the sshfs ./data
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-cranberry}"
REMOTE_REPO="${REMOTE_REPO:-robustness}"                                   # home-relative on cranberry
REMOTE_SCRATCH="${REMOTE_SCRATCH:-/local/scratch/a/agarapat}"              # my own writable scratch dir
REMOTE_DATA_DIR="${REMOTE_DATA_DIR:-$REMOTE_SCRATCH/robustness-data}"      # data/ target on scratch

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RSYNC=(rsync -az)   # portable across macOS rsync + GNU rsync
# Exclude the top-level data entry (anchored '/data') so a code push never transfers data/ or the symlink.
CODE_EXCL=(--exclude '.git/' --exclude '.venv/' --exclude '__pycache__/' --exclude '.DS_Store'
           --exclude '.pytest_cache/' --exclude '.mypy_cache/' --exclude '.ruff_cache/' --exclude '*.egg-info/'
           --exclude '.ipynb_checkpoints/' --exclude '/data')

push_code()  { echo ">> push code          local -> $REMOTE_HOST:$REMOTE_REPO";
               "${RSYNC[@]}" --delete "${CODE_EXCL[@]}" "$SCRIPT_DIR/" "$REMOTE_HOST:$REMOTE_REPO/"; }
push_input() { echo ">> push data/input    local -> $REMOTE_HOST (additive; lands on scratch via the symlink)";
               "${RSYNC[@]}" "$SCRIPT_DIR/data/input/" "$REMOTE_HOST:$REMOTE_DATA_DIR/input/"; }
pull_results() { echo ">> pull output+cache  $REMOTE_HOST -> local data/ (mirror; skip if you use sshfs)";
               mkdir -p "$SCRIPT_DIR/data/output" "$SCRIPT_DIR/data/cache";
               "${RSYNC[@]}" --delete "$REMOTE_HOST:$REMOTE_DATA_DIR/output/" "$SCRIPT_DIR/data/output/";
               "${RSYNC[@]}" --delete "$REMOTE_HOST:$REMOTE_DATA_DIR/cache/" "$SCRIPT_DIR/data/cache/"; }

# One-time: create the scratch data dirs, the ownership README, and the ~/robustness/data symlink.
setup() {
  echo ">> setup scratch data + symlink on $REMOTE_HOST"
  ssh "$REMOTE_HOST" "
    set -euo pipefail
    owner=\$(stat -c '%U' '$REMOTE_SCRATCH' 2>/dev/null || echo '?')
    [ \"\$owner\" = \"\$(whoami)\" ] || { echo \"REFUSING: $REMOTE_SCRATCH not owned by me (owner=\$owner)\"; exit 1; }
    mkdir -p '$REMOTE_DATA_DIR'/input '$REMOTE_DATA_DIR'/cache '$REMOTE_DATA_DIR'/output
    mkdir -p '$REMOTE_REPO'
    cd '$REMOTE_REPO'
    if [ -L data ]; then echo \"  data symlink -> \$(readlink data)\";
    elif [ -e data ]; then echo '  WARN: data exists and is not a symlink'; ls -ld data;
    else ln -s '$REMOTE_DATA_DIR' data; echo '  created data symlink'; fi
    ls -ld data; df -h '$REMOTE_DATA_DIR' | tail -1
  "
  echo "Done. data/ lives on scratch; nothing under it counts against \$HOME."
}

# Build/refresh the standardized conda+uv env on cranberry (idempotent). No hand-managed venv.
env_build() {
  echo ">> build conda+uv env on $REMOTE_HOST (scratch-backed caches; \$HOME quota stays clear)"
  ssh "$REMOTE_HOST" "REMOTE_SCRATCH='$REMOTE_SCRATCH' REMOTE_REPO='$REMOTE_REPO' bash -s" <<'EOF'
set -euo pipefail
MF="$REMOTE_SCRATCH/miniforge3"
export CONDA_ENVS_PATH="$REMOTE_SCRATCH/envs"
export CONDA_PKGS_DIRS="$REMOTE_SCRATCH/conda-pkgs"
export UV_CACHE_DIR="$REMOTE_SCRATCH/uv-cache"
export PIP_CACHE_DIR="$REMOTE_SCRATCH/pip-cache"
export TMPDIR="$REMOTE_SCRATCH/tmp"; mkdir -p "$TMPDIR" "$REMOTE_SCRATCH/envs"
cd "$REMOTE_REPO"
if [ ! -x "$MF/bin/conda" ]; then
  curl -fsSL -o "$REMOTE_SCRATCH/mf.sh" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash "$REMOTE_SCRATCH/mf.sh" -b -p "$MF"; rm -f "$REMOTE_SCRATCH/mf.sh"
fi
source "$MF/etc/profile.d/conda.sh"
conda config --append envs_dirs "$REMOTE_SCRATCH/envs" 2>/dev/null || true  # so `conda activate robustness` works by name
[ -x "$REMOTE_SCRATCH/envs/robustness/bin/python" ] || conda env create -f environment.yml
conda activate robustness
echo "python: $(python --version)  uv: $(uv --version)  ruff: $(ruff --version)"
uv sync --frozen
# presto is installed --no-deps (upstream pins torch==2.0/numpy==1.23.5/einops==0.6.0/... and would
# wreck the resolved env); the project already provides the deps presto needs at runtime.
uv pip install --no-deps "git+https://github.com/nasaharvest/presto.git@11e207a668a34336ced1d8e492a1bd5849b96c4a"
rm -rf src/*.egg-info *.egg-info 2>/dev/null || true  # editable install works via the .pth; don't leave egg-info behind
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())"
echo "ENV READY. Activate with: source $MF/etc/profile.d/conda.sh && conda activate robustness"
EOF
}

# sshfs-mount cranberry's data dir onto local ./data (so the Mac sees data/ without storing it).
do_mount() {
  if ! command -v sshfs >/dev/null 2>&1; then
    echo "sshfs not found. Install it once (FUSE-T is easiest -- no kernel extension, no reboot):"
    echo "  brew install fuse-t fuse-t-sshfs"
    echo "Alternative (macFUSE -- needs admin approval of a kernel extension + a reboot):"
    echo "  brew install --cask macfuse   # approve the kext in System Settings, then REBOOT"
    echo "  brew install gromgit/fuse/sshfs-mac"
    exit 1
  fi
  if mount | grep -q " $SCRIPT_DIR/data "; then echo "already mounted: $SCRIPT_DIR/data"; exit 0; fi
  if [ -e "$SCRIPT_DIR/data" ] && [ -n "$(ls -A "$SCRIPT_DIR/data" 2>/dev/null)" ]; then
    echo "REFUSING: local ./data is non-empty (would shadow real files). Move/remove it first."; ls -A "$SCRIPT_DIR/data"; exit 1
  fi
  mkdir -p "$SCRIPT_DIR/data"
  echo ">> sshfs $REMOTE_HOST:$REMOTE_DATA_DIR -> $SCRIPT_DIR/data"
  sshfs "$REMOTE_HOST:$REMOTE_DATA_DIR" "$SCRIPT_DIR/data" \
    -o reconnect,follow_symlinks,defer_permissions,noappledouble,volname=robustness-data
  echo "mounted. ls data/:"; ls "$SCRIPT_DIR/data"
}

do_umount() {
  echo ">> unmount $SCRIPT_DIR/data"
  umount "$SCRIPT_DIR/data" 2>/dev/null || diskutil unmount "$SCRIPT_DIR/data" 2>/dev/null || {
    echo "not mounted, or busy"; exit 1; }
  echo "unmounted."
}

case "${1:-push}" in
  setup)  setup ;;
  env)    env_build ;;
  push)   push_code ;;
  input)  push_code; push_input ;;
  pull)   pull_results ;;
  mount)  do_mount ;;
  umount|unmount) do_umount ;;
  *) echo "usage: ./sync.sh [setup|env|push|input|pull|mount|umount]"; exit 1 ;;
esac
echo "Done."
