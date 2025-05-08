import sys
import os
import platform
sys.path.append('../../../src')

import logging
import datetime
# Set logging level to suppress warnings
logging.getLogger().setLevel(logging.ERROR)
# Just disables the warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json

import conf
import argparse

from botopeninglead import BotLead
from sample import Sample
from objects import Card
from auction import VULN
import numpy as np
from colorama import Fore, init

init()


SEATS = ['north', 'east', 'south', 'west']

def lead(obj, models, sampler, dds, verbose):
    if obj['contract'] is None:
        return None

    dealer_i = 'NESW'.index(obj["dealer"])
        
    decl_i = 'NESW'.index(obj['contract'][-1])
    lead_i = (decl_i + 1) % 4

    hand_lead = obj[SEATS[lead_i]]
    bot = BotLead(VULN[obj['vuln']], hand_lead, models, sampler, lead_i, dealer_i, dds, verbose)
    lead_card_indexes, _ = bot.get_opening_lead_candidates(obj['auction'])
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

    args = parser.parse_args()

    sys.stderr.write(f'NS = {args.bidder}\n')
    
    np.set_printoptions(precision=2, suppress=True)

    sys.stderr.write(f'{Fore.CYAN}{datetime.datetime.now():%Y-%m-%d %H:%M:%S} Loading configuration. Python {platform.python_version()} Python {platform.python_version()}{Fore.RESET}\n')  

    configuration = conf.load(args.bidder)

    try:
        if configuration["models"]['tf_version'] == "2":
            sys.stderr.write("Loading version 2\n")
            from nn.models_tf2 import Models
        else: 
            # Default to version 1. of Tensorflow
            from nn.models_tf2 import Models
    except KeyError:
            # Default to version 1. of Tensorflow
            from nn.models_tf2 import Models

    models = Models.from_conf(configuration,"../../..")
    sampler = Sample.from_conf(configuration)

    from ddsolver import ddsolver
    dds = ddsolver.DDSolver()

    for line in sys.stdin:
        obj = json.loads(line)
        obj['lead'] = lead(obj, models, sampler, dds, False)

        print(json.dumps(obj))
        sys.stdout.flush()
