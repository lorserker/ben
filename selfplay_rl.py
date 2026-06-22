# selfplay_rl.py




import os
import sys
import logging
import datetime
import time
import warnings

print("SCRIPT FILE:", __file__)
print("WORKING DIR:", os.getcwd())


# --- Environment / paths ---
sys.path.append(os.path.join(os.environ["BEN_HOME"], "src"))
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Point to BEN src


# Suppress noisy logs
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
import absl.logging
absl.logging.get_absl_handler().python_handler.stream = open(os.devnull, 'w')
absl.logging.set_verbosity(absl.logging.FATAL)
absl.logging.set_stderrthreshold(absl.logging.FATAL)

import argparse
import numpy as np
import tensorflow as tf

import conf
from botbidder import BotBid
from bidding import bidding
from sample import Sample
from ddsolver import ddsolver

# your deal generator module
import deck52  # used by your generator
from collections import deque

VULN = {
    'None': [False, False],
    'N-S': [True, False],
    'E-W': [False, True],
    'Both': [True, True]
}


def generate_deals(n_boards):
    dealer = list('NESW')
    vuln = deque(['None', 'N-S', 'E-W', 'Both'])

    for i in range(n_boards):
        deal_str = deck52.random_deal()

        if i % 4 == 0 and i > 0:
            vuln.append(vuln.popleft())

        yield dealer[i % 4], vuln[i % 4], deal_str


def bid_hand(hands, dealer, vuln, models_ns_ew, samplers, dds, verbose):
    dealer_i = 'NESW'.index(dealer)

    bidder_bots = [
        BotBid(
            VULN[vuln],
            hand,
            models_ns_ew[i % 2],
            samplers[i % 2],
            i,
            dealer_i,
            dds,
            False,
            verbose
        )
        for i, hand in enumerate(hands)
    ]

    auction = ['PAD_START'] * dealer_i
    turn_i = dealer_i

    bid_responses = []
    while not bidding.auction_over(auction):
        if verbose:
            print("_________________________________")
            print("Turn", "NESW"[turn_i], hands[turn_i])
        bid_resp = bidder_bots[turn_i].bid(auction)
        bid_responses.append(bid_resp)
        auction.append(bid_resp.bid)
        turn_i = (turn_i + 1) % 4

    return auction, bid_responses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bidder", type=str, required=True,
                        help="Path to BEN config (used for both NS and EW)")
    parser.add_argument("--boards", type=int, default=1000,
                        help="Number of random boards to self-play")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    verbose = args.verbose

    sys.stderr.write(
        f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} selfplay_rl.py - TF {tf.__version__}\n"
    )

    configuration = conf.load(args.bidder)

    try:
        if configuration["models"]['tf_version'] == "2":
            from nn.models_tf2 import Models
        else:
            from nn.models_tf2 import Models
    except KeyError:
        from nn.models_tf2 import Models

    base_path = os.getenv("BEN_HOME") or os.getcwd()

    models_ns = Models.from_conf(configuration, base_path)
    models_ew = Models.from_conf(configuration, base_path)

    sampler_ns = Sample.from_conf(configuration)
    sampler_ew = Sample.from_conf(configuration)

    dds = ddsolver.DDSolver()

    t_job_start = time.time()
    for idx, (dealer, vuln, hands_str) in enumerate(generate_deals(args.boards), start=1):
        # optional: reproducible per-board randomness
        

        hands = deck52.reorder_hand(hands_str).split()



        if verbose:
            sys.stderr.write(f"Bidding board {idx} dealer={dealer} vuln={vuln}\n")

        bid_start = time.time()
        auction, bid_responses = bid_hand(
            hands,
            dealer,
            vuln,
            [models_ns, models_ew],
            [sampler_ns, sampler_ew],
            dds,
            verbose
        )
        if verbose:
            sys.stderr.write(
                f"Board {idx} auction: {' '.join(auction)} in {time.time() - bid_start:.2f}s\n"
            )

        # IMPORTANT:
        # RL data collection happens *inside* BotBid.bid()
        # via your RL block using candidates[0].expected_imp etc.
        # This script just drives boards; RLWriter writes to disk.

    sys.stderr.write(
        f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} "
        f"Self-play {args.boards} boards in {time.time() - t_job_start:.2f}s\n"
    )


if __name__ == "__main__":
    main()
