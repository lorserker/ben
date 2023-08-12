import sys
sys.path.append('../../../src')

import json

import conf
import argparse

from nn.models import Models
from bots import BotLead
from sample import Sample
from objects import Card
from auction import VULN


SEATS = ['north', 'east', 'south', 'west']

def lead(obj, models, ns, ew, sampler):
    if obj['contract'] is None:
        return None
    
    decl_i = 'NESW'.index(obj['contract'][-1])
    lead_i = (decl_i + 1) % 4

    hand_lead = obj[SEATS[lead_i]]
    bot = BotLead(VULN[obj['vuln']], hand_lead, models, ns, ew, sampler)
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
    
    models = Models.from_conf(conf.load(args.bidder))
    sampler = Sample.from_conf(conf.load(args.bidder))
    for line in sys.stdin:
        obj = json.loads(line)
        obj['lead'] = lead(obj, models, ns, ew, sampler)

        print(json.dumps(obj))
        sys.stdout.flush()
