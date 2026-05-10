"""Print the BEN release version, parsed from ../src/game.py.

Used by BuildAll.cmd to name the release archives, e.g. MvsM-0.8.7.7.zip.
Prints 'unknown' (and exits 0) if it can't find the version line.
"""
import os
import re

_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "game.py")
try:
    _m = re.search(r"^\s*version\s*=\s*['\"]([^'\"]+)['\"]", open(_src, encoding="utf-8").read(), re.M)
    print(_m.group(1) if _m else "unknown")
except OSError:
    print("unknown")
