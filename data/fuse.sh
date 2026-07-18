#!/usr/bin/env bash
#
# fuse.sh — (re)mount the robustness sshfs / macFUSE mounts onto this Mac.
#
# This script lives at data/fuse.sh; the mounts it manages live under data/output/:
#   output/machines/<box>   for cranberry, dewberry, avocado, eggplant, gilbreth
#   output/collated         for the merge-hub (now on gilbreth, moved off cranberry)
#
# Safe to re-run: mounts that are already alive are left untouched; dead or
# stale ones are force-unmounted and remounted. All mounts use reconnect +
# keepalive. avocado is mounted read-only on purpose.
#
# NOTE on gilbreth: Purdue RCAC requires MFA on every connect, so mounting it
# prompts for MFA (approve the push) and it CANNOT auto-reconnect if the SSH
# session drops — just re-run `./fuse.sh gilbreth` to bring it back.
#
# Usage:
#   ./fuse.sh                  # (re)mount every configured mount
#   ./fuse.sh cranberry gilbreth   # only the named ones
#   ./fuse.sh -f [names...]     # force: remount even if currently alive
#   ./fuse.sh -u [names...]     # unmount the named ones (or all)
#   ./fuse.sh -s               # status only (liveness of each), no changes
#   ./fuse.sh -h               # this help
#
set -uo pipefail

SSHFS="/usr/local/bin/sshfs"
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # this script's dir = .../data
# The mountpoints are NOT beside this script -- they live under data/output/, which .gitignore
# excludes. Deriving them from a separate root rather than from BASE means moving the script
# cannot silently relocate the mounts (which is exactly what moving it here would have done).
MOUNT_ROOT="$BASE/output"
COMMON="reconnect,ServerAliveInterval=15,ServerAliveCountMax=3"

# name | remote (host:path) | mountpoint (relative to MOUNT_ROOT) | extra sshfs -o opts
MOUNTS=(
  "cranberry|cranberry:/local/scratch/a/agarapat/robustness-run-data/cranberry|machines/cranberry|auto_cache,volname=cranberry"
  "dewberry|dewberry:/local/scratch/a/agarapat/robustness-run-data/dewberry|machines/dewberry|auto_cache,volname=dewberry"
  "avocado|avocado:/local/scratch/a/agarapat/robustness-run-data/avocado|machines/avocado|ro,defer_permissions,volname=robustness-avocado"
  "eggplant|eggplant:/local/scratch/a/agarapat/robustness-run-data/eggplant|machines/eggplant|auto_cache,volname=eggplant"
  "gilbreth|gilbreth:/scratch/gilbreth/agarapat/robustness/data|machines/gilbreth|auto_cache,volname=gilbreth"
  "collated|gilbreth:/scratch/gilbreth/agarapat/robustness/data/results|collated|auto_cache,volname=collated"
  # digital-ag is not reachable through the ECN jump; set its real remote path
  # and uncomment to enable (left disabled so a run can't hang on it):
  # "digital-ag|digital-ag:/PATH/TO/robustness-run-data|machines/digital-ag|auto_cache,volname=digital-ag"
)

mounted()      { mount | grep -q " $1 "; }
# alive = actually in the mount table AND responsive (a bare empty mountpoint
# dir is listable but NOT a live mount, so ls alone is not enough).
alive()        { mounted "$1" && timeout 8 ls "$1" >/dev/null 2>&1; }
force_umount() { timeout 20 umount -f "$1" 2>/dev/null || timeout 20 diskutil unmount force "$1" >/dev/null 2>&1; }

do_mount() {  # name remote mp opts force
  local name="$1" remote="$2" mp="$3" opts="$4" force="$5"
  if alive "$mp"; then
    if [ "$force" = 1 ]; then echo "[$name] alive -> forcing remount"; else echo "[$name] already alive"; return 0; fi
  fi
  mounted "$mp" && force_umount "$mp"
  mkdir -p "$mp"
  echo "[$name] mounting $remote"
  if ! $SSHFS -o "$COMMON,$opts" "$remote" "$mp"; then echo "[$name] FAILED (sshfs error)"; return 1; fi
  sleep 2
  if alive "$mp"; then echo "[$name] OK (alive)"; else echo "[$name] FAILED (not responding)"; return 1; fi
}

do_umount() {  # name mp
  local name="$1" mp="$2"
  if mounted "$mp"; then force_umount "$mp"; echo "[$name] unmounted"; else echo "[$name] not mounted"; fi
}

MODE=mount; FORCE=0; ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    -f|--force)             FORCE=1 ;;
    -u|--umount|--unmount)  MODE=umount ;;
    -s|--status)            MODE=status ;;
    -h|--help)              grep '^#' "$0" | sed 's/^#\{1,\} \{0,1\}//'; exit 0 ;;
    -*)                     echo "unknown option: $1" >&2; exit 2 ;;
    *)                      ARGS+=("$1") ;;
  esac
  shift
done

want() {  # selected? (no ARGS = all)
  [ ${#ARGS[@]} -eq 0 ] && return 0
  local n; for n in "${ARGS[@]}"; do [ "$n" = "$1" ] && return 0; done; return 1
}

rc=0
for entry in "${MOUNTS[@]}"; do
  IFS='|' read -r name remote rel opts <<< "$entry"
  want "$name" || continue
  mp="$MOUNT_ROOT/$rel"
  case "$MODE" in
    mount)  do_mount  "$name" "$remote" "$mp" "$opts" "$FORCE" || rc=1 ;;
    umount) do_umount "$name" "$mp" ;;
    status) alive "$mp" && echo "[$name] ALIVE" || echo "[$name] DEAD" ;;
  esac
done
exit $rc
