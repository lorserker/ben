#!/usr/bin/env bash
# stopservers.sh - stop all BEN servers started by runservers.sh
# (appserver, appserverold, gameapi, gameserver). macOS/Linux.
#
# Usage:  bash stopservers.sh
PATTERN='gameapi\.py|gameserver\.py|appserver\.py|appserverold\.py'

if ! pgrep -f "$PATTERN" >/dev/null 2>&1; then
    echo "No BEN servers running."
    exit 0
fi

echo "Stopping these BEN processes:"
pgrep -fl "$PATTERN"

# Graceful shutdown first, then force-kill anything still alive.
pkill -TERM -f "$PATTERN" 2>/dev/null
sleep 2
if pgrep -f "$PATTERN" >/dev/null 2>&1; then
    echo "Force-killing survivors ..."
    pkill -KILL -f "$PATTERN" 2>/dev/null
    sleep 1
fi

if pgrep -f "$PATTERN" >/dev/null 2>&1; then
    echo "WARNING: some processes are still running:"
    pgrep -fl "$PATTERN"
    exit 1
fi
echo "All BEN servers stopped."
