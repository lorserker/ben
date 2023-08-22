import sys
sys.path.append('../../../src')

import argparse
import json

import conf
import copy

import numpy as np

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


def bid_hand(hands, dealer, vuln, models_ns_ew, ns, ew, do_search, nntrust):

    if (nntrust==None) :
        nntrust = 0.1
    dealer_i = 'NESW'.index(dealer)
    
    bidder_bots = [BotBid(VULN[vuln], hand, models_ns_ew[i % 2], ns, ew, nntrust, False) for i, hand in enumerate(hands)]

    auction = ['PAD_START'] * dealer_i

    do_search_NS = do_search == "NS" or do_search == "Both"
    do_search_EW = do_search == "EW"or do_search == "Both"

    turn_i = dealer_i

    while not bidding.auction_over(auction):
        if do_search_NS and ((turn_i ) % 2 == 0) :
            bid = bidder_bots[turn_i].bid(auction).bid
            auction.append(bid)
        else:
            if do_search_EW and ((turn_i ) % 2 == 1) :
                bid = bidder_bots[turn_i].bid(auction).bid
                auction.append(bid)
            else:
                # to be able to recreate the board from the gameserver we let the bot bid, but ignores the result
                bot = copy.copy(bidder_bots[turn_i])
                bid = bot.bid(auction).bid
                candidates = bidder_bots[turn_i].get_bid_candidates(auction)
                #print(f"Candidate: {candidates[0].bid}")
                bid = candidates[0].bid
                auction.append(bid)
        
        turn_i = (turn_i + 1) % 4  # next player's turn
    
    return auction

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--bidderNS', type=str)
    parser.add_argument('--bidderEW', type=str)
    parser.add_argument('--set', type=str)
    parser.add_argument('--search', type=str)
    parser.add_argument('--nntrust', type=float)

    args = parser.parse_args()
    
    sys.stderr.write(f'NS = {args.bidderNS}\n')
    sys.stderr.write(f'EW = {args.bidderEW}\n')
    sys.stderr.write(f'search = {args.search}\n')
    sys.stderr.write(f'nn trust = {args.nntrust}\n')
    
    models_ns = Models.from_conf(conf.load(args.bidderNS))
    models_ew = Models.from_conf(conf.load(args.bidderEW))


    for index, line in enumerate(open(args.set)):        
        # To make the boards reproducable random is seeded at the beginning of each board
        np.random.seed(42)

        parts = line.strip().split()
        dealer = parts[0]
        vuln = parts[1]
        hands = parts[2:]

        ns = 0
        ew = 0
        #sys.stderr.write(f'Bidding board {index + 1}\n')
        auction = bid_hand(hands, dealer, vuln, [models_ns, models_ew], ns, ew, args.search, args.nntrust)

        record = {
            'board' : index + 1,
            'contract': bidding.get_contract(auction),
            'dealer': dealer,
            'vuln': vuln,
            'north': hands[0],
            'east': hands[1],
            'south': hands[2],
            'west': hands[3],
            'auction': auction
        }

        print(json.dumps(record))
        sys.stdout.flush()
