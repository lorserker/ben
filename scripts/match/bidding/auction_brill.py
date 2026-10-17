"""Like auction.py, but one side bids with Brill through a Brill.Service /bid endpoint.

python auction_brill.py --brill=NS --bidder=BEN-21GF.conf --set=set0001.txt [--brill-url=http://localhost:5200] [--opponent-system=BEN] > auctions.json

The output has the same format as auction.py, so lead.py, score.py and compare.py work unchanged.
"""
import sys
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

sys.path.append('../../../src')

import logging
import warnings
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

import argparse
import datetime
import json
import time
import urllib.parse
import urllib.request

import numpy as np

import conf
from botbidder import BotBid
from bidding import bidding
from sample import Sample

VULN = {
    'None': [False, False],
    'N-S': [True, False],
    'E-W': [False, True],
    'Both': [True, True]
}


class BrillBidder:
    """Asks Brill.Service for the call of one seat."""

    def __init__(self, url, hand, seat, dealer, vuln, opponent_system=None):
        self.url, self.hand, self.seat, self.dealer, self.vuln = url, hand, seat, dealer, vuln
        self.opponent_system = opponent_system

    def bid(self, auction):
        calls = [{'PASS': 'P'}.get(b, b) for b in auction if b != 'PAD_START']
        params = {'hand': self.hand, 'seat': self.seat, 'dealer': self.dealer, 'vul': self.vuln, 'ctx': '-'.join(calls)}
        if self.opponent_system:
            # Brill reads the opponents' calls through this system instead of its own
            params['ewSystem' if self.seat in 'NS' else 'nsSystem'] = self.opponent_system
        query = urllib.parse.urlencode(params)
        with urllib.request.urlopen(f'{self.url}/bid?{query}', timeout=120) as response:
            call = json.loads(response.read())['bid'].upper().replace('NT', 'N')
        call = {'P': 'PASS', 'DBL': 'X', 'RDBL': 'XX'}.get(call, call)
        if not bidding.can_bid(call, auction):
            raise ValueError(f'Brill returned illegal call {call} after {calls}')
        return call


def bid_hand(hands, dealer, vuln, brill_side, models, sampler, dds, url, opponent_system=None):
    dealer_i = 'NESW'.index(dealer)
    bidders = []
    for i, hand in enumerate(hands):
        if 'NESW'[i] in brill_side:
            bidders.append(BrillBidder(url, hand, 'NESW'[i], dealer, vuln, opponent_system))
        else:
            bidders.append(BotBid(VULN[vuln], hand, models, sampler, i, dealer_i, dds, False, False))

    auction = ['PAD_START'] * dealer_i
    turn_i = dealer_i
    while not bidding.auction_over(auction):
        bidder = bidders[turn_i]
        auction.append(bidder.bid(auction) if isinstance(bidder, BrillBidder) else bidder.bid(auction).bid)
        turn_i = (turn_i + 1) % 4
    return auction


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--brill', choices=['NS', 'EW'], required=True, help='The side Brill bids for')
    parser.add_argument('--bidder', type=str, required=True, help='BEN configuration for the other side')
    parser.add_argument('--set', type=str, required=True)
    parser.add_argument('--brill-url', default='http://localhost:5200')
    parser.add_argument('--opponent-system', default=None, help='Brill system to read the BEN side through, e.g. BEN')
    args = parser.parse_args()

    sys.stderr.write(f'Brill = {args.brill} via {args.brill_url}\nBEN = {args.bidder}\n')
    configuration = conf.load(args.bidder)
    from nn.models_tf2 import Models
    models = Models.from_conf(configuration, "../../..")
    sampler = Sample.from_conf(configuration)
    from ddsolver import ddsolver
    dds = ddsolver.DDSolver()

    t_job_start = time.time()
    for index, line in enumerate(open(args.set)):
        # Same seeding as auction.py, so the BEN side bids as it would there
        np.random.seed(42)
        parts = line.strip().split()
        dealer, vuln, hands = parts[0], parts[1], parts[2:]
        sys.stderr.write(f'Bidding board {index + 1}\n')
        auction = bid_hand(hands, dealer, vuln, args.brill, models, sampler, dds, args.brill_url, args.opponent_system)
        print(json.dumps({
            'board': index + 1,
            'contract': bidding.get_contract(auction),
            'dealer': dealer,
            'vuln': vuln,
            'north': hands[0],
            'east': hands[1],
            'south': hands[2],
            'west': hands[3],
            'auction': auction
        }))
        sys.stdout.flush()

    sys.stderr.write(f'{datetime.datetime.now():%Y-%m-%d %H:%M:%S} Set bid in {time.time() - t_job_start:0.0f} seconds.\n')
