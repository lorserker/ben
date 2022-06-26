import sys
sys.path.append('../../../src')

import argparse
import json

import conf

from nn.bidder import Bidder
from nn.models import Models
from bots import BotBid
from bidding import bidding

VULN = {
    'None': [False, False],
    'N-S': [True, False],
    'E-W': [False, True],
    'Both': [True, True]
}


def bid_hand(hands, dealer, vuln, models_ns_ew, do_search):
    dealer_i = 'NESW'.index(dealer)
    
    bidder_bots = [BotBid(VULN[vuln], hand, models_ns_ew[i % 2]) for i, hand in enumerate(hands)]

    auction = ['PAD_START'] * dealer_i

    turn_i = dealer_i

    while not bidding.auction_over(auction):
        if do_search:
            bid = bidder_bots[turn_i].bid(auction).bid
            auction.append(bid)
        else:
            candidates = bidder_bots[turn_i].get_bid_candidates(auction)
            bid = candidates[0].bid
            auction.append(bid)
        
        turn_i = (turn_i + 1) % 4  # next player's turn
    
    return auction

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--bidderNS', type=str)
    parser.add_argument('--bidderEW', type=str)
    parser.add_argument('--set', type=str)
    parser.add_argument('--search', action='store_true')

    args = parser.parse_args()
    
    sys.stderr.write(f'NS = {args.bidderNS}\n')
    sys.stderr.write(f'EW = {args.bidderEW}\n')
    sys.stderr.write(f'search = {args.search}\n')
    
    models_ns = Models.from_conf(conf.load(args.bidderNS))
    models_ew = Models.from_conf(conf.load(args.bidderEW))

    for line in open(args.set):
        parts = line.strip().split()
        dealer = parts[0]
        vuln = parts[1]
        hands = parts[2:]

        auction = bid_hand(hands, dealer, vuln, [models_ns, models_ew], args.search)

        record = {
            'dealer': dealer,
            'vuln': vuln,
            'north': hands[0],
            'east': hands[1],
            'south': hands[2],
            'west': hands[3],
            'auction': auction,
            'contract': bidding.get_contract(auction)
        }

        print(json.dumps(record))
        sys.stdout.flush()
