#!/usr/bin/env bash
# runservers.sh - macOS/Linux equivalent of runservers.cmd.
#
# Launches the BEN app/API/game servers in the background (staggered, like the
# Windows version's 10s gaps so TensorFlow model loads don't all hit at once),
# each logging to ../logs/<name>.log. Uses the repo venv (../.venv) if present.
#
# Usage:
#   cd ben/src
#   bash runservers.sh
#
# Stop everything:
#   pkill -f 'gameapi.py|gameserver.py|appserver.py|appserverold.py'
set -u

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../ben/src
cd "$SRC"

# Pick the interpreter: an activated venv first, then the repo's ../.venv,
# then python3 (macOS often has no bare 'python'), then python.
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PY="$VIRTUAL_ENV/bin/python"
elif [ -x "$SRC/../.venv/bin/python" ]; then
    PY="$SRC/../.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PY="$(command -v python)"
else
    echo "No python found. Create the venv first: python3.12 -m venv ../.venv && source ../.venv/bin/activate && pip install -r ../requirements.txt" >&2
    exit 1
fi
echo "Using Python: $("$PY" -c 'import sys; print(sys.executable, sys.version.split()[0])')"

LOGS="$SRC/../logs"
mkdir -p "$LOGS"

# launch <logname> <workdir> <script> [args...]
launch() {
    local name="$1"; local wd="$2"; shift 2
    echo "=> starting $name"
    ( cd "$wd" && nohup "$PY" "$@" >"$LOGS/$name.log" 2>&1 & )
    sleep 10
}

# --- Web UI app server (frontend/appserver.py, default port 8080) ---
launch appserver       "$SRC/frontend" appserver.py --host 0.0.0.0
launch appserverold    "$SRC"          appserverold.py --port 8081 --host 0.0.0.0

# --- Play API (used from BBO) ---
# NOTE: port 80 is privileged on macOS/Linux. Either run this script with sudo,
#       change --port 80 to e.g. 8080, or front it with a reverse proxy.
#       Comment the next line out if you don't expose it externally.
launch api-bbo-80      "$SRC"          gameapi.py --config config/GIB-BBO.conf --port 80 --host 0.0.0.0 --allowed-hosts "*"
launch api-bbo-8085    "$SRC"          gameapi.py --config config/GIB-BBO.conf            --host 0.0.0.0 --allowed-hosts "*"

# --- Game servers (websocket), one per bidding model/config ---
# NOTE: gameserver.py already binds 0.0.0.0 (it's a raw websocket server with no
#       Host-header check), so it accepts neither --host nor --allowed-hosts.
launch ben-21gf-4440   "$SRC"          gameserver.py --config config/BEN-21GF.conf --port 4440
launch ben-sayc-4441   "$SRC"          gameserver.py --config config/BEN-SAYC.conf --port 4441
launch gib-bbo-4442    "$SRC"          gameserver.py --config config/GIB-BBO.conf  --port 4442
launch default-4443    "$SRC"          gameserver.py --config config/default.conf  --port 4443

echo
echo "All instances launched. Logs: $LOGS"
echo "Check one:   tail -f $LOGS/default-4443.log"
echo "Stop all:    pkill -f 'gameapi.py|gameserver.py|appserver.py|appserverold.py'"
