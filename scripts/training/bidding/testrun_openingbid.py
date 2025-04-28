import sys
sys.path.append('../../../src')
import argparse
import logging
import os
import conf
import scoring
import compare
import datetime
import time
from ddsolver.ddsolver import DDSolver
from colorama import Fore, Back, Style, init

# Set logging level to suppress warnings
logging.getLogger().setLevel(logging.ERROR)
# Just disables the warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

import numpy as np

from bidding import bidding
from sample import Sample
from botbidder import BotBid

DD = DDSolver(dds_mode=1)


def get_par(hands, vulnerable):
    hands_pbn = 'N:' + ' '.join(hands)    
    par_score = DD.calculatepar(hands_pbn, vulnerable, False)
    return par_score

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

def get_hcp(hand):
    hcp = {'A': 4, 'K': 3, 'Q': 2, 'J': 1}
    return sum([hcp.get(c, 0) for c in hand])


def main():
    sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Testrun validating bidding.\n')
    
    # Record the start time
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Process bidding data using a bidder model.")
    parser.add_argument("config_path", help="Path to the configuration file")
    parser.add_argument("filename", help="Path to the input filename containing hands and dealer information")
    parser.add_argument("--alternate", type=bool, default=False, help="Skip the closed room board")
    parser.add_argument("--verbose", type=bool, default=False, help="Print extra information")
    parser.add_argument("--nocolor", type=bool, default=False, help="Remove color from output")
    args = parser.parse_args()

    config_path = args.config_path
    filename = args.filename
    extension = os.path.splitext(filename)[1].lower() 
    if extension == ".pbn":
        print("Expecting a file with hand and bids on 2 lines")
        sys.exit()
    verbose = args.verbose
    alternate = args.alternate
    color = not args.nocolor

    np.set_printoptions(precision=2, suppress=True, linewidth=1200,threshold=np.inf)

    config = conf.load(config_path)
   
    sys.stderr.write(f"Loading tensorflow {tf.__version__}\n")
    sys.stderr.write(f"NumPy Version : {np.__version__}\n")
    try:
        if (config["models"]['tf_version'] == "2"):
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


    models = Models.from_conf(config,"../../..")
    sys.stderr.write("Models loaded\n")
    sampler = Sample.from_conf(config,"../../..")

    sys.stderr.write("Configuration loaded\n")
    with open(filename, 'r') as input_file:

        lines = input_file.readlines()
        # Remove comments at the beginning of the file
        lines = [line.strip().replace("*",'') for line in lines if not line.strip().startswith('#')]        
        n = len(lines) // 2
        matching = 0
        different = 0
        rights = {}
        wrongs = {}
        auctions = []
        sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Loaded {n} deals\n')
        for i in range(n):
            if color: print(Fore.BLUE, end='')
            if alternate and i % 2 == 1:
                continue
            parts = lines[i*2].strip().split()
            hands = parts[:]
            parts = lines[i*2+1].strip().replace("  "," ").split()
            #if not ( parts[2:][0] != 'P'):
            #    continue
            #print("Board: ",i+1)
            dealer_i = 'NESW'.index(parts[0])
            vuln = {'N-S': (True, False), 'E-W': (False, True), 'None': (False, False), 'Both': (True, True)}
            vuln_ns, vuln_ew = vuln[parts[1]]
            bidder_bots = [BotBid([vuln_ns, vuln_ew], hand, models, sampler, i, dealer_i, DD, verbose) for i, hand in enumerate(hands)]
    
            auction = ['PAD_START'] * dealer_i

            turn_i = dealer_i
            
            candidates, _ = bidder_bots[turn_i].get_bid_candidates(auction)
            if len(candidates) > 0:
                bid = candidates[0].bid
                auction.append(bid)
                turn_i = (turn_i + 1) % 4  # next player's turn

            if color: print(Fore.WHITE, end='')
            #print(" ".join(parts[:2]), " ".join(hands))
            auction_str = " ".join(auction).replace('PAD_START ', '')
            trained_bidding = " ".join(parts[2:3]).replace('P',"PASS")
            if (trained_bidding != auction_str):
                if candidates[0].bid != "PASS":
                    if color: print(Fore.RED, end='')
                else:
                    if candidates[0].insta_score < 0.8:
                        if color: print(Fore.CYAN, end='')
                #print(f"{parts[2:3][0]:4} {get_hcp(hands[dealer_i]):2.0f} {hands[dealer_i]} {candidates[0].bid:4} {candidates[0].insta_score:.3f} {vuln}")
                auctions.append(f'{parts[2:3][0]:4} {get_hcp(hands[dealer_i]):2.0f} {hands[dealer_i]} {candidates[0].bid:4} {candidates[0].insta_score:.3f} {"VULN" if vuln[parts[1]][dealer_i % 2] else "NO"} {i+1}')
                different += 1
                key = parts[2:3][0]
                print("key: ", key)
                if candidates[0].bid != "PASS":
                    key = key + "->" + candidates[0].bid
                    if key in wrongs:
                        wrongs[key] += 1
                    else:
                        wrongs.update({key: 1})
                else:
                    if key in wrongs:
                        wrongs[key] += 1
                    else:
                        wrongs.update({key: 1})
                        rights.update({key: 0})
            else: 
                matching += 1
                if parts[2:3][0] in rights:
                    rights[parts[2:3][0]] += 1
                else:
                    rights.update({parts[2:3][0]: 1})
                    wrongs.update({parts[2:3][0]: 0})
                continue

        print(rights, wrongs)

        unique_sorted_keys = sorted(set(auctions))

        #print(auctions)
        # Print all unique keys sorted
        for key in unique_sorted_keys:
            print(key)
        if color: print(Fore.GREEN)
        sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Matched {matching} deals\n')
        print(rights)
        if color: print(Fore.YELLOW)
        sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} different {different} deals\n')
        print(wrongs)
        # Record the end time
        end_time = time.time()

        # Calculate the elapsed time in seconds
        execution_time = end_time - start_time

        if color: print(Fore.WHITE)
        # Display the result
        sys.stderr.write(f"Execution time: {execution_time:.2f} seconds\n")
        if color: print(Fore.RESET)
                        

if __name__ == '__main__':
    main()

