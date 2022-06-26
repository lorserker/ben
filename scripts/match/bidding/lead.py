import sys
sys.path.append('../../../src')

import json

import conf

from nn.models import Models
from bots import BotLead
from objects import Card
from auction import VULN


SEATS = ['north', 'east', 'south', 'west']

def get_models():
    models = Models.from_conf(conf.load('../../../default.conf'))
    return models

def lead(obj, models):
    if obj['contract'] is None:
        return None
    
    decl_i = 'NESW'.index(obj['contract'][-1])
    lead_i = (decl_i + 1) % 4

    hand_lead = obj[SEATS[lead_i]]
    bot = BotLead(VULN[obj['vuln']], hand_lead, models)
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

    models = get_models()

    for line in sys.stdin:
        obj = json.loads(line)
        obj['lead'] = lead(obj, models)

        print(json.dumps(obj))
        sys.stdout.flush()
