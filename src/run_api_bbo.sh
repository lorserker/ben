#!/usr/bin/env bash
# run_api_bbo.sh - start the Play API (gameapi.py) with config/GIB-BBO.conf.
#
# Unlike runservers.sh this runs ONE server, in the foreground, and refuses to
# start on the wrong interpreter rather than falling back to whatever `python3`
# happens to be first on PATH. That fallback is the cause of two confusing
# failures on macOS: a `dds3` ImportError (the vendored extension is built for
# one Python version), and a PIMC DDS backend that loads as an error string and
# aborts the process partway through the first trick.
#
# Port 80 is the default because that is what runservers.sh exposes as
# 'api-bbo-80' and what the reverse proxy forwards to. Binding it needs root.
#
# Usage:
#   cd ben/src
#   sudo bash run_api_bbo.sh            # port 80, same as production
#   PORT=8085 bash run_api_bbo.sh       # unprivileged port for a local test
#   bash run_api_bbo.sh --verbose true  # extra args are passed to gameapi.py
#
# Stop with Ctrl-C. For all servers in the background, use runservers.sh.
set -eu

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../ben/src
cd "$SRC"

CONFIG="config/GIB-BBO.conf"
PORT="${PORT:-80}"

# Fail here rather than letting Python raise PermissionError halfway through
# loading the models.
if [ "$PORT" -lt 1024 ] && [ "$(id -u)" -ne 0 ]; then
    echo "Port $PORT is privileged and this is not root." >&2
    echo "  sudo bash run_api_bbo.sh        # production port" >&2
    echo "  PORT=8085 bash run_api_bbo.sh   # unprivileged, local testing only" >&2
    exit 1
fi

# Pick the interpreter: an activated venv, else the repo's ../.venv. No bare
# `python3` fallback on purpose - see the note above.
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PY="$VIRTUAL_ENV/bin/python"
elif [ -x "$SRC/../.venv/bin/python" ]; then
    PY="$SRC/../.venv/bin/python"
else
    cat >&2 <<'EOF'
No BEN virtualenv found.

Expected one of:
  - an activated venv ($VIRTUAL_ENV), or
  - ../.venv/bin/python next to this checkout

Create it with a Python matching the vendored dds3 extension (3.12):

  cd ..
  python3.12 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt

Starting BEN with the system python3 is not supported: the compiled dds3
extension will not import, and PIMC's DDS backend may fail to resolve.
EOF
    exit 1
fi

echo "Interpreter: $("$PY" -c 'import sys; print(sys.executable, sys.version.split()[0])')"
echo "Config:      $CONFIG"
echo "Port:        $PORT"
echo

# Fail fast with a readable diagnosis instead of a traceback mid-startup.
if ! "$PY" preflight.py --config "$CONFIG"; then
    echo >&2
    echo "Pre-flight failed - not starting gameapi.py." >&2
    exit 1
fi
echo

exec "$PY" gameapi.py --config "$CONFIG" --port "$PORT" --host 0.0.0.0 --allowed-hosts '*' "$@"
