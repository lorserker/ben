import sys
sys.path.append('D:/github/ben/src')
import argparse
import logging
import os
import conf
import scoring
import datetime
import time
from ddsolver.ddsolver import DDSolver

# Set logging level to suppress warnings
logging.getLogger().setLevel(logging.ERROR)
# Just disables the warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

from bidding import bidding
from sample import Sample
from bots import BotBid

DD = DDSolver(dds_mode=1)

def get_dd_score(hands, contract, is_vulnerable):
    if contract is None:
        return 0

    strain_i = 'NSHDC'.index(contract[1])
    decl_i = 'NESW'.index(contract[-1])
    hands_pbn = 'N:' + ' '.join(hands)
    sol = DD.solve(strain_i, (decl_i + 1) % 4, [], [hands_pbn],3)
    dd_tricks = 13 - max(vals[0] for vals in sol.values())
    dd_score = scoring.score(contract, is_vulnerable, n_tricks=dd_tricks)
    dd_score = dd_score if decl_i % 2 == 0 else -dd_score

    return dd_score


def get_models(bidder_model_path):
    models = Models(None, None, None, None, None)
    models.bidder_model = Bidder('bidder', bidder_model_path)
    return models


def main():
    sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Testrun validating bidding.\n')
    
    # Record the start time
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Process bidding data using a bidder model.")
    parser.add_argument("config_path", help="Path to the configuration file")
    parser.add_argument("filename", help="Path to the input filename containing hands and dealer information")
    parser.add_argument("--alternate", type=bool, default=False, help="Skip the closed room board")
    parser.add_argument("--verbose", type=bool, default=False, help="Print extra information")
    args = parser.parse_args()

    config_path = args.config_path
    filename = args.filename
    extension = os.path.splitext(filename)[1].lower() 
    if extension == ".pbn":
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
        # Remove comments at the beginning of the file
        lines = [line.strip().replace("*",'') for line in lines if not line.strip().startswith('#')]        
        n = len(lines) // 2
        matching = 0
        better = 0
        worse = 0
        same = 0
        sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Loaded {n} deals\n')
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
            bidder_bots = [BotBid([vuln_ns, vuln_ew], hand, models, sampler, i, dealer_i, verbose) for i, hand in enumerate(hands)]
    
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
            auction_str = " ".join(auction).replace('PAD_START ', '')
            trained_bidding = " ".join(parts[2:]).replace('P',"PASS")
            if (trained_bidding != auction_str):
                #trained_auction = [bidding.BID2ID[bid] for bid in trained_bidding.split(' ')]
                trained_auction = [bid for bid in trained_bidding.split(' ')]
                trained_auction = ['PAD_START'] * dealer_i + trained_auction
                contract = bidding.get_contract(trained_auction, dealer_i, models)
                vuln = False
                #print("Contract:",contract)
                dd_score_before= get_dd_score(hands, contract, vuln)
                print(" ".join(parts[:2]), " ".join(hands))
                contract = bidding.get_contract(auction, dealer_i, models)
                vuln = False
                dd_score_now = get_dd_score(hands, contract, vuln)
                print(" ".join(parts[2:]), dd_score_before)
                print(auction_str.replace('PASS','P'), dd_score_now)
                if dd_score_before == dd_score_now:
                    same += 1
                else:
                    if dd_score_now > dd_score_before:
                        better += 1
                    else:
                        worse += 1
            else: 
                #print(" ".join(hands))
                #print(auction_str)
                matching += 1
        sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Matched {matching} deals\n')
        sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} better {better} deals\n')
        sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} worse {worse} deals\n')
        sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} same score {same} deals\n')
        # Record the end time
        end_time = time.time()

        # Calculate the elapsed time in seconds
        execution_time = end_time - start_time

        # Display the result
        print(f"Execution time: {execution_time:.2f} seconds")
                

if __name__ == '__main__':
    main()
