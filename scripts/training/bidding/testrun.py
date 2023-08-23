import sys
sys.path.append('../../../src')
import conf
import argparse

from collections import Counter

from nn.bidder import Bidder
from nn.models import Models
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
    parser.add_argument("--ns", type=int, default=-1, help="System for NS")
    parser.add_argument("--ew", type=int, default=-1, help="System for EW")
    args = parser.parse_args()

    config_path = args.config_path
    filename = args.filename
    ns = args.ns
    ew = args.ew

    models = Models.from_conf(conf.load(config_path))

    sampler = Sample.from_conf(conf.load(config_path))
    with open(filename, 'r') as input_file:
        for line in input_file:
            parts = line.strip().split()
            hands = parts[2:]
            dealer_i = 'NESW'.index(parts[0])
            
            #Set vulnerable based on input in file
            #Read NS and EW system from conf-file
            bidder_bots = [BotBid([False, False], hand, models, ns, ew,-1, sampler, False) for hand in hands]

            auction = ['PAD_START'] * dealer_i

            turn_i = dealer_i

            while not bidding.auction_over(auction):
                candidates = bidder_bots[turn_i].get_bid_candidates(auction)
                if len(candidates) > 0:
                    bid = candidates[0].bid
                    auction.append(bid)
                    turn_i = (turn_i + 1) % 4  # next player's turn
                else:
                    auction.append("??")
                    break

            auction_str = "-".join(auction).replace("PASS", "P").replace('PAD_START-', '')
            print(f'{line.strip()} {auction_str}')


if __name__ == '__main__':
    main()
