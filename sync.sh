#!/usr/bin/env bash
# One-command sync for the robustness repo.
#
# Source-of-truth split:
#   - CODE + data/input/  live on LOCAL  (you edit/stage here, they flow UP)
#   - output + cache live on digital-ag (runs write them, they flow DOWN)
# So a code push never touches the server's results, and a results pull never
# touches your code.
#
# Disk note: on digital-ag $HOME is small/full, so runs write output + cache to a big
# scratch disk ($REMOTE_SCRATCH, default /var/tmp/robustness) by setting ROBUSTNESS_SCRATCH.
# Pulls read from there. Locally there is no scratch redirect, so results land in data/.
#
# Run on LOCAL (the usual case):
#   ./sync.sh           push code up, then pull results (output + cache) down
#   ./sync.sh push      only push code up
#   ./sync.sh pull      only pull results down
#   ./sync.sh input     push code up AND push data/input/ up  (run after staging a new dataset)
#   ./sync.sh setup     create the remote scratch dir + print the export line to use before a run
#
# Run on digital-ag (optional; needs LOCAL_REPO = an SSH path back to this Mac repo):
#   ./sync.sh           push output + cache back to LOCAL_REPO
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-digital-ag}"
REMOTE_REPO="${REMOTE_REPO:-~/robustness}"
REMOTE_SCRATCH="${REMOTE_SCRATCH:-/var/tmp/robustness}"   # big-disk home for output + cache on the server
LOCAL_REPO="${LOCAL_REPO:-}"          # only needed for the digital-ag -> Mac direction

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="$(hostname)"
is_digital_ag() { [[ "$HOST" == *digital-ag* ]] || [[ "$(hostname -f 2>/dev/null || true)" == *digital-ag* ]]; }

RSYNC=(rsync -avz)   # portable across macOS openrsync + GNU rsync (no --info=progress2)
CODE_EXCL=(--exclude '.git/' --exclude '.venv/' --exclude '__pycache__/' --exclude '.DS_Store'
           --exclude '.pytest_cache/' --exclude '.mypy_cache/' --exclude '.ipynb_checkpoints/'
           --exclude 'data/')                       # data/ is synced separately, by direction
# Large + machine-local cache artifacts that should NOT cross machines: geotessera tiles
# (re-downloadable) and the TESSERA model weights (each box downloads its own copy).
CACHE_EXCL=(--exclude 'global_0.1_degree*' --exclude 'tessera/')

push_code()    { echo ">> push code            local -> $REMOTE_HOST";
                 "${RSYNC[@]}" --delete "${CODE_EXCL[@]}" "$SCRIPT_DIR/" "$REMOTE_HOST:$REMOTE_REPO/"; }
push_input()   { echo ">> push data/input      local -> $REMOTE_HOST  (additive)";
                 "${RSYNC[@]}" "$SCRIPT_DIR/data/input/" "$REMOTE_HOST:$REMOTE_REPO/data/input/"; }
pull_results() { echo ">> pull output+cache    $REMOTE_HOST:$REMOTE_SCRATCH -> local data/  (mirror)";
                 mkdir -p "$SCRIPT_DIR/data/output" "$SCRIPT_DIR/data/cache";
                 "${RSYNC[@]}" --delete "$REMOTE_HOST:$REMOTE_SCRATCH/output/" "$SCRIPT_DIR/data/output/";
                 "${RSYNC[@]}" --delete "${CACHE_EXCL[@]}" "$REMOTE_HOST:$REMOTE_SCRATCH/cache/" "$SCRIPT_DIR/data/cache/"; }
setup()        { echo ">> create remote scratch dirs on $REMOTE_HOST: $REMOTE_SCRATCH/{output,cache}";
                 ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_SCRATCH/output' '$REMOTE_SCRATCH/cache'";
                 echo "   On the server, export this before launching a run (e.g. in ~/.bashrc):";
                 echo "     export ROBUSTNESS_SCRATCH=$REMOTE_SCRATCH"; }

if is_digital_ag; then
  if [[ -z "$LOCAL_REPO" ]]; then
    echo "ERROR: LOCAL_REPO is not set (needed to push results back to your Mac)."
    echo '  export LOCAL_REPO="you@your-mac:/Users/akshithchowdary/Developer/Projects/org/abe/robustness"'
    exit 1
  fi
  SCRATCH="${ROBUSTNESS_SCRATCH:-$SCRIPT_DIR/data}"   # where this run wrote output + cache
  echo ">> on digital-ag: push output+cache ($SCRATCH) -> $LOCAL_REPO/data"
  "${RSYNC[@]}" --delete "$SCRATCH/output/" "$LOCAL_REPO/data/output/"
  "${RSYNC[@]}" --delete "${CACHE_EXCL[@]}" "$SCRATCH/cache/" "$LOCAL_REPO/data/cache/"
  echo "Done."
  exit 0
fi

case "${1:-all}" in
  push)  push_code ;;
  pull)  pull_results ;;
  input) push_code; push_input ;;
  setup) setup ;;
  all)   push_code; pull_results ;;
  *) echo "usage: ./sync.sh [push|pull|input|setup]"; exit 1 ;;
esac
echo "Done."
