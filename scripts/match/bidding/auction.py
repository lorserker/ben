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


def bid_hand(hands, dealer, vuln, models_ns_ew, samplers, verbose):

    dealer_i = 'NESW'.index(dealer)
    
    bidder_bots = [BotBid(VULN[vuln], hand, models_ns_ew[i % 2], samplers[i % 2], i, dealer_i, verbose) for i, hand in enumerate(hands)]

    turn_i = dealer_i

    while not bidding.auction_over(auction):
        bid = bidder_bots[turn_i].bid(auction).bid
        auction.append(bid)
        turn_i = (turn_i + 1) % 4  # next player's turn
    
    return auction

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--bidderNS', type=str)
    parser.add_argument('--bidderEW', type=str)
    parser.add_argument('--set', type=str)

    args = parser.parse_args()

    sys.stderr.write(f'NS = {args.bidderNS}\n')
    sys.stderr.write(f'EW = {args.bidderEW}\n')

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

        sys.stderr.write(f'Bidding board {index + 1}\n')
        auction = bid_hand(hands, dealer, vuln, [models_ns, models_ew], [Sample.from_conf(configuration_ns), Sample.from_conf(configuration_ew)],False)

        record = {
            'board' : index + 1,
            'contract': bidding.get_contract(auction, 'NESW'.index(dealer), models_ns),
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
