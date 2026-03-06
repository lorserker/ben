#!/bin/bash
export PATH="$HOME/.dotnet:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export DOTNET_ROOT="$HOME/.dotnet"
export BBA_BIN="/mnt/d/test/BBA/bin/Debug/net9.0"
cd /mnt/d/GitHub/ben/src/bba
python3 test_bid.py
