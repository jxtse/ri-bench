#!/usr/bin/env bash
# Run DeepSeek Harness (dsh, headless profile) as a read-confined senior agent for RI-Bench.
# Kernel-level confinement via macOS Seatbelt: file reads allowed only under the case workspace (cwd),
# the harness runtime, the node toolchain, DSH_HOME and the system trees; everything else under
# /Users and /Volumes is denied. Writes allowed only to DSH_HOME (session log) and temp areas.
# dsh's own fs-sandbox only fences writes ("reads always pass through"), so this wrapper is what
# makes the "no reading other folders" guarantee hold for this harness.
set -euo pipefail
WS="$(pwd -P)"
HARNESS_REAL="$(cd "${DSH_REPO:-${DSH_HARNESS_DIR:-$HOME/Documents/GitHub/deepseek-harness}}" && pwd -P)"
HARNESS_LINK="${DSH_REPO:-${DSH_HARNESS_DIR:-$HOME/Documents/GitHub/deepseek-harness}}"
: "${DSH_HOME:?DSH_HOME must be set (isolated profile home)}"
DSH_HOME_REAL="$(cd "$DSH_HOME" && pwd -P)"
NODE_BIN="$(command -v node)"; NODE_REAL="$(cd "$(dirname "$(readlink -f "$NODE_BIN")")"/.. && pwd -P)"
PNPM_STORE="${HOME}/Library/pnpm"
PROFILE="$(mktemp -t ribench-dsh-sb).sb"
cat > "$PROFILE" <<EOF
(version 1)
(allow default)
(deny network-outbound (remote unix-socket))
(deny file-read* file-write*
  (subpath "/Users")
  (subpath "/Volumes"))
(allow file-read*
  (subpath "$WS")
  (subpath "$HARNESS_REAL")
  (subpath "$HARNESS_LINK")
  (subpath "$DSH_HOME_REAL")
  (subpath "$DSH_HOME")
  (subpath "$NODE_REAL")
  (subpath "/opt/homebrew")
  (subpath "$PNPM_STORE"))
; ancestor directories: metadata + readlink only (path resolution), never their contents
(allow file-read-metadata
$(p="$WS"; while [ "$p" != "/" ]; do echo "  (literal \"$p\")"; p="$(dirname "$p")"; done)
$(p="$HARNESS_REAL"; while [ "$p" != "/" ]; do echo "  (literal \"$p\")"; p="$(dirname "$p")"; done)
$(p="$HARNESS_LINK"; while [ "$p" != "/" ]; do echo "  (literal \"$p\")"; p="$(dirname "$p")"; done)
$(p="$DSH_HOME_REAL"; while [ "$p" != "/" ]; do echo "  (literal \"$p\")"; p="$(dirname "$p")"; done)
  (literal "$HOME"))
(allow file-write*
  (subpath "$DSH_HOME_REAL")
  (subpath "$DSH_HOME")
  (subpath "/private/tmp")
  (subpath "/private/var/folders")
  (subpath "/dev"))
EOF
trap 'rm -f "$PROFILE"' EXIT
export HOME="$DSH_HOME_REAL"   # keep node/dsh from touching the real home
exec /usr/bin/sandbox-exec -f "$PROFILE" "$NODE_BIN" "$HARNESS_LINK/apps/cli/lib/bin.js" --profile headless "$@"
