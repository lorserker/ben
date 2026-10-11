#!/usr/bin/env python3
"""
Report names that are used but never defined anywhere under src/.

An undefined name is invisible until the one code path that touches it runs:
the module imports, compiles and the service starts. Two examples this caught:
an import missing on the opening-lead path only, and `merged_card_resp` in the
discard branch of botcardplayer.py. pyflakes finds them without running BEN.

    python scripts/check-undefined-names.py          # check src/
    python scripts/check-undefined-names.py path ... # check other files or folders

Exits 1 if any undefined name is found. Only undefined names are reported -
unused imports and the like are ignored. Needs pyflakes: pip install pyflakes
"""
import os
import sys

try:
    from pyflakes import api, messages, reporter
except ImportError:
    print("pyflakes is not installed: pip install pyflakes")
    sys.exit(2)

UNDEFINED = (messages.UndefinedName, messages.UndefinedLocal, messages.UndefinedExport)


class UndefinedOnly(reporter.Reporter):
    def __init__(self):
        super().__init__(sys.stdout, sys.stderr)
        self.found = []
        self.star_imports = []

    def flake(self, message):
        if isinstance(message, UNDEFINED):
            self.found.append(message)
        elif isinstance(message, messages.ImportStarUsage) and message.filename not in self.star_imports:
            self.star_imports.append(message.filename)


def main():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
    paths = sys.argv[1:] or [os.path.normpath(root)]
    rep = UndefinedOnly()
    api.checkRecursive(paths, rep)

    for message in rep.found:
        print(message)
    for filename in rep.star_imports:
        print(f"note: {filename} uses 'from ... import *', so undefined names there cannot be detected")
    print(f"{len(rep.found)} undefined name(s)")
    return 1 if rep.found else 0


if __name__ == "__main__":
    sys.exit(main())
