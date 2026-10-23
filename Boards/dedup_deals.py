"""Write the deals of a merged PBN once each (same cards, dealer and vulnerability),
with only Board, Dealer, Vulnerable and Deal, ready for any system to bid.

Usage:
    python dedup_deals.py All.PBN deals.pbn
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bid_with_epbot import read_boards  # noqa: E402

if len(sys.argv) != 3:
    sys.exit(__doc__)

n = 0
with open(sys.argv[2], 'w', encoding='utf-8', newline='\n') as out:
    out.write('% PBN 2.1\n% EXPORT\n\n')
    for dealer, vulnerable, nesw in read_boards(sys.argv[1]):
        n += 1
        out.write(f'[Board "{n}"]\n[Dealer "{dealer}"]\n[Vulnerable "{vulnerable}"]\n'
                  f'[Deal "N:{" ".join(nesw)}"]\n\n')
print(f"{n} deals -> {sys.argv[2]}")
