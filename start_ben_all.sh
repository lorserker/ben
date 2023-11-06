#!/bin/bash

cd "$(dirname "$0")"/src

# is it a bug?
if [ ! -d models ]; then
    ln -s ../models models
fi

# this is all in one wrapper script mainly for container
python3 gameserver.py & # listen on 4443 for websocket
python3 appserver.py --host 0.0.0.0 &  # listen on 8080 for browser

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?