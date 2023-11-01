import sys
import os
sys.path.append('../../../src')

import logging

# Set logging level to suppress warnings
logging.getLogger().setLevel(logging.ERROR)
# Just disables the warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import json

import conf
import copy

import numpy as np

from bots import BotBid
from bidding import bidding
from sample import Sample

VULN = {
    'None': [False, False],
    'N-S': [True, False],
    'E-W': [False, True],
    'Both': [True, True]
}


def bid_hand(hands, dealer, vuln, models_ns_ew, ns, ew, do_search, samplers, verbose):

    dealer_i = 'NESW'.index(dealer)
    
    bidder_bots = [BotBid(VULN[vuln], hand, models_ns_ew[i % 2], ns, ew, samplers[i % 2], verbose) for i, hand in enumerate(hands)]

    auction = ['PAD_START'] * dealer_i

    do_search_ns = do_search == "NS" or do_search == "Both"
    do_search_ew = do_search == "EW"or do_search == "Both"

    turn_i = dealer_i

    while not bidding.auction_over(auction):
        if do_search_ns and ((turn_i ) % 2 == 0) :
            bid = bidder_bots[turn_i].bid(auction).bid
            auction.append(bid)
        else:
            if do_search_ew and ((turn_i ) % 2 == 1) :
                bid = bidder_bots[turn_i].bid(auction).bid
                auction.append(bid)
            else:
                # to be able to recreate the board from the gameserver we let the bot bid, but ignores the result
                bot = copy.copy(bidder_bots[turn_i])
                bid = bot.bid(auction).bid
                candidates, passout = bidder_bots[turn_i].get_bid_candidates(auction)
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

    args = parser.parse_args()

    sys.stderr.write(f'NS = {args.bidderNS}\n')
    sys.stderr.write(f'EW = {args.bidderEW}\n')
    sys.stderr.write(f'search = {args.search}\n')

    configuration_ns = conf.load(args.bidderNS)
    configuration_ew = conf.load(args.bidderEW)

    np.set_printoptions(precision=2, suppress=True)

    try:
        if configuration_ns["models"]['tf_version'] == "2":
            sys.stderr.write("Loading version 2\n")
            from nn.models_tf2 import Models
        else: 
            # Default to version 1. of Tensorflow
            from nn.models import Models
    except KeyError:
            # Default to version 1. of Tensorflow
            from nn.models import Models

    models_ns = Models.from_conf(configuration_ns,"..\..\..")
    models_ew = Models.from_conf(configuration_ns,"..\..\..")

    for index, line in enumerate(open(args.set)):        
        # To make the boards reproducable random is seeded at the beginning of each board
        np.random.seed(42)

        parts = line.strip().split()
        dealer = parts[0]
        vuln = parts[1]
        hands = parts[2:]

        ns = -1
        ew = -1
        sys.stderr.write(f'Bidding board {index + 1}\n')
        auction = bid_hand(hands, dealer, vuln, [models_ns, models_ew], ns, ew, args.search, [Sample.from_conf(configuration_ns), Sample.from_conf(configuration_ew)],False)

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
