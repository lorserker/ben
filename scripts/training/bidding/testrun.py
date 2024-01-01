import sys
sys.path.append('../../../src')
import argparse
import logging
import os
import conf

# Set logging level to suppress warnings
logging.getLogger().setLevel(logging.ERROR)
# Just disables the warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

from sample import Sample
from bots import BotBid
from bidding import bidding


def get_models(bidder_model_path):
    models = Models(None, None, None, None, None)
    models.bidder_model = Bidder('bidder', bidder_model_path)
    return models


def main():
    parser = argparse.ArgumentParser(description="Process bidding data using a bidder model.")
    parser.add_argument("config_path", help="Path to the configuration file")
    parser.add_argument("filename", help="Path to the input filename containing hands and dealer information")
    parser.add_argument("--alternate", type=bool, default=False, help="Skip the closed room board")
    parser.add_argument("--verbose", type=bool, default=False, help="Print extra information")
    args = parser.parse_args()

    config_path = args.config_path
    filename = args.filename
    extension = os.path.splitext(filename)[1].lower() 
    if extension == ".pbn" or extension == ".ben":
        print("Expecting a file with hand and bids on 2 lines")
        sys.exit()
    verbose = args.verbose
    alternate = args.alternate

    np.set_printoptions(precision=2, suppress=True, linewidth=220)

    config = conf.load(config_path)
   
    try:
        if (config["models"]['tf_version'] == "2"):
            print("Loading version 2")
            from nn.bidder_tf2 import Bidder
            from nn.models_tf2 import Models
        else: 
            # Default to version 1. of Tensorflow
            from nn.bidder import Bidder
            from nn.models import Models
    except KeyError:
            # Default to version 1. of Tensorflow
            from nn.bidder import Bidder
            from nn.models import Models


    models = Models.from_conf(config,"..\..\..")
    sampler = Sample.from_conf(config,"..\..\..")

    with open(filename, 'r') as input_file:

        lines = input_file.readlines()
        n = len(lines) // 2
        matching = 0
        print(f"Loaded {n} deals")
        for i in range(n):
            print("Board: ",i+1)
            if alternate and i % 2 == 1:
                continue
            parts = lines[i*2].strip().split()
            hands = parts[:]
            parts = lines[i*2+1].strip().replace("  "," ").split()
            dealer_i = 'NESW'.index(parts[0])
            vuln = {'N-S': (True, False), 'E-W': (False, True), 'None': (False, False), 'Both': (True, True)}
            vuln_ns, vuln_ew = vuln[parts[1]]
            bidder_bots = [BotBid([vuln_ns, vuln_ew], hand, models, sampler, verbose) for hand in hands]

            auction = ['PAD_START'] * dealer_i

            turn_i = dealer_i

            while not bidding.auction_over(auction):
                candidates, passout = bidder_bots[turn_i].get_bid_candidates(auction)
                if len(candidates) > 0:
                    bid = candidates[0].bid
                    auction.append(bid)
                    turn_i = (turn_i + 1) % 4  # next player's turn
                else:
                    auction.append("??")
                    break

            auction_str = " ".join(auction).replace("PASS", "P").replace('PAD_START ', '')
            trained_bidding = " ".join(parts[2:])
            if (trained_bidding != auction_str):
                print(" ".join(hands))
                print(" ".join(parts[2:]))
                print(auction_str)
            else: 
                print(" ".join(hands))
                print(auction_str)
                matching += 1
        print(matching," boards matched")
                

if __name__ == '__main__':
    main()
