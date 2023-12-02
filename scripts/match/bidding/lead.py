import sys
import os
sys.path.append('../../../src')

import logging

# Set logging level to suppress warnings
logging.getLogger().setLevel(logging.ERROR)
# Just disables the warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json

import conf
import argparse

from bots import BotLead
from sample import Sample
from objects import Card
from auction import VULN
import numpy as np


SEATS = ['north', 'east', 'south', 'west']

def lead(obj, models, ns, ew, sampler, verbose):
    if obj['contract'] is None:
        return None
    
    decl_i = 'NESW'.index(obj['contract'][-1])
    lead_i = (decl_i + 1) % 4

    hand_lead = obj[SEATS[lead_i]]
    bot = BotLead(VULN[obj['vuln']], hand_lead, models, sampler, verbose)
    lead_card_indexes, _ = bot.get_lead_candidates(obj['auction'])
    lead_card_i = lead_card_indexes[0]
    suit_i = lead_card_i // 8
    rank_i = lead_card_i % 8
    if rank_i < 7:
        return Card.from_code(lead_card_i, xcards=True).symbol()
    else:
        suit = hand_lead.split('.')[suit_i]
        return f'{"SHDC"[suit_i]}{suit[-1]}'


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--bidder', type=str)
    parser.add_argument("--ns", type=int, default=-1, help="System for NS")
    parser.add_argument("--ew", type=int, default=-1, help="System for EW")

    args = parser.parse_args()

    ns = args.ns
    ew = args.ew

    sys.stderr.write(f'NS = {args.bidder}\n')
    
    np.set_printoptions(precision=2, suppress=True)

    configuration = conf.load(args.bidder)

    try:
        if configuration["models"]['tf_version'] == "2":
            sys.stderr.write("Loading version 2\n")
            from nn.models_tf2 import Models
        else: 
            # Default to version 1. of Tensorflow
            from nn.models import Models
    except KeyError:
            # Default to version 1. of Tensorflow
            from nn.models import Models

    models = Models.from_conf(configuration,"..\..\..")
    sampler = Sample.from_conf(configuration)
    for line in sys.stdin:
        obj = json.loads(line)
        obj['lead'] = lead(obj, models, ns, ew, sampler, False)

        print(json.dumps(obj))
        sys.stdout.flush()
