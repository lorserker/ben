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

import numpy as np

from bidding import bidding
from sample import Sample

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


def main():
    sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Badcontracts validating bidding.\n')
    
    # Record the start time
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Process bidding data using a bidder model.")
    parser.add_argument("filename", help="Path to the input filename containing hands and dealer information")
    parser.add_argument("--alternate", type=bool, default=False, help="Skip the closed room board")
    parser.add_argument("--verbose", type=bool, default=False, help="Print extra information")
    parser.add_argument("--nocolor", type=bool, default=False, help="Remove color from output")
    args = parser.parse_args()

    filename = args.filename
    extension = os.path.splitext(filename)[1].lower() 
    if extension == ".pbn":
        print("Expecting a file with hand and bids on 2 lines")
        sys.exit()
    verbose = args.verbose
    alternate = args.alternate
    color = not args.nocolor

    np.set_printoptions(precision=2, suppress=True, linewidth=1200,threshold=np.inf)

    with open(filename, 'r') as input_file:

        lines = input_file.readlines()
        # Remove comments at the beginning of the file
        lines = [line.strip().replace("*",'') for line in lines if not line.strip().startswith('#')]        
        n = len(lines) // 2
        matching = 0
        gameswing = 0
        worse = 0
        removed = 0
        bad_deals = []
        save_deals = []
        sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Loaded {n} deals\n')
        for i in range(n):
            if color: print(Fore.BLUE, end='')
            sys.stderr.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Board: {i+1}\n")
            if alternate and i % 2 == 1:
                continue
            parts = lines[i*2].strip().split()
            hands = parts[:]
            parts = lines[i*2+1].strip().replace("  "," ").split()
            dealer_i = 'NESW'.index(parts[0])
            vuln = {'N-S': (True, False), 'E-W': (False, True), 'None': (False, False), 'Both': (True, True)}
            vuln_ns, vuln_ew = vuln[parts[1]]
    
            auction = " ".join(parts[2:]).replace('*','')
            contract = bidding.get_contract(['PAD_START'] * dealer_i +auction.split(' '))
            if not contract == None:
                declarer = contract[-1]
                if declarer == "N" or declarer == "S":
                    vuln = vuln_ns
                else :
                    vuln = vuln_ew
                dd_score_now = get_dd_score(hands, contract, vuln)
            else:
                dd_score_now = 0
            par_score = get_par(hands, [vuln_ns, vuln_ew])
            if abs(dd_score_now - par_score) <= 100:
                matching += 1
                if color: print(Fore.GREEN, end='')
                sys.stderr.write(f"{auction} {contract} {par_score} {dd_score_now}\n")
                save_deals.append(lines[i*2] + '\n' + lines[i*2+1] + "\n")
            else:
                if abs(dd_score_now - par_score) <= 500:
                    if color: print(Fore.BLUE, end='')
                    gameswing += 1
                    sys.stderr.write(f"{auction} {contract} {par_score} {dd_score_now}\n")
                if abs(dd_score_now - par_score) > 500 and abs(dd_score_now - par_score) <= 1000:
                    if color: print(Fore.YELLOW, end='')
                    worse += 1
                    sys.stderr.write(f"{auction} {contract} {par_score} {dd_score_now}\n")
                if abs(dd_score_now - par_score) > 1000:
                    if color: print(Fore.RED, end='')
                    removed += 1
                    sys.stderr.write(f"{auction} {contract} {par_score} {dd_score_now}\n")
                    bad_deals.append(lines[i*2] + '\n' + lines[i*2+1] + "\n")
                else:
                    save_deals.append(lines[i*2] + '\n' + lines[i*2+1] + "\n")
                    
            


        if color: print(Fore.GREEN)
        sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Matched {matching} deals\n')
        if color: print(Fore.BLUE)
        sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} gameswing {gameswing} deals\n')
        if color: print(Fore.YELLOW)
        sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} worse {worse} deals\n')
        if color: print(Fore.RED)
        sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Slamswing {removed} deals\n')
        # Record the end time
        end_time = time.time()

        # Calculate the elapsed time in seconds
        execution_time = end_time - start_time

        if color: print(Fore.WHITE)
        # Display the result
        sys.stderr.write(f"Execution time: {execution_time:.2f} seconds\n")
        if color: print(Fore.RESET)
                        
        with open('bad.ben', 'w', encoding='utf-8') as file:  # Open the output file with UTF-8 encoding
            for board in bad_deals:
                file.write(board)
            print("File bad.ben generated")
        with open('valid.ben', 'w', encoding='utf-8') as file:  # Open the output file with UTF-8 encoding
            for board in save_deals:
                file.write(board)
            print("File valid.ben generated")


if __name__ == '__main__':
    main()

