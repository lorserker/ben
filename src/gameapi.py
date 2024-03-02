from gevent import monkey

from bots import BotBid, BotLead, CardPlayer
from bidding import bidding
from objects import Card
import deck52
monkey.patch_all()

from bottle import Bottle, run, static_file, redirect, template, request, response
from bottle_session import SessionPlugin

import bottle
import json
import os
import logging

# Set logging level to suppress warnings
logging.getLogger().setLevel(logging.ERROR)
# Just disables the warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# This import is only to help PyInstaller when generating the executables
import tensorflow as tf

import uuid
import shelve
import time
import datetime
import asyncio
import websockets
import argparse
import game
import human
import conf
import functools
import os
import numpy as np
from websockets.exceptions import ConnectionClosedOK
from sample import Sample
from urllib.parse import parse_qs, urlparse
from pbn2ben import load

def get_execution_path():
    # Get the directory where the program is started from either PyInstaller executable or the script
    return os.getcwd()

random = True
# For some strange reason parameters parsed to the handler must be an array
board_no = []
seed = None
board_no.append(0) 

# Get the path to the config file
config_path = get_execution_path()
    
base_path = os.getenv('BEN_HOME') or config_path

parser = argparse.ArgumentParser(description="Game API")
parser.add_argument("--host", default="localhost", help="Hostname for appserver")
parser.add_argument("--config", default=f"{base_path}/config/default.conf", help="Filename for configuration")
parser.add_argument("--verbose", type=bool, default=False, help="Output samples and other information during play")
parser.add_argument("--port", type=int, default=8085, help="Port for appserver")

args = parser.parse_args()

configfile = args.config
verbose = args.verbose
port = args.port
np.set_printoptions(precision=2, suppress=True, linewidth=240)

configuration = conf.load(configfile)

try:
    if (configuration["models"]['tf_version'] == "2"):
        print("Loading version 2")
        from nn.models_tf2 import Models
    else: 
        # Default to version 1. of Tensorflow
        from nn.models import Models
except KeyError:
        # Default to version 1. of Tensorflow
        from nn.models import Models

models = Models.from_conf(configuration, base_path.replace(os.path.sep + "src",""))
sampler = Sample.from_conf(configuration, False)

print('models loaded')

host = args.host
print(f'http://{host}:{port}/home')

app = Bottle()

plugin = SessionPlugin(cookie_lifetime=600)
app.install(plugin)

# CORS middleware
class CorsPlugin(object):
    name = 'cors'
    api = 2

    def apply(self, callback, route):
        def wrapper(*args, **kwargs):
            response.headers['Access-Control-Allow-Origin'] = '*'  # Replace * with your allowed domain if needed
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'
            if request.method != 'OPTIONS':
                return callback(*args, **kwargs)
        return wrapper

app.install(CorsPlugin())

class CardPlayerPool:
    def __init__(self):
        self.player_pool = {}

    def get_player(self, username):
        # Retrieve the player object from the pool
        return self.player_pool.get(username)

    def set_player(self, username, player):
        # Set the player object in the pool
        self.player_pool[username] = player

    def remove_player(self, username):
        # Remove the player object from the pool
        if username in self.player_pool:
            del self.player_pool[username]
            
player_pool = CardPlayerPool()

class Player:
    def __init__(self, user):
        self.user = user
        self.trick_i = 0
        self.cardplayer_i = -1
        self.leader_i = 0
        self.player_cards_played = [[] for _ in range(4)]
        self.shown_out_suits = [set() for _ in range(4)]
        self.dealer_i = 0
        self.current_trick = []
        self.current_trick52 = []
        # Initialize other attributes as needed

@app.route('/')
def default():
    html = '<h1><a href="/app/bridge.html?S=x&A=2&T=2">Play Now</a></h1>\n'
    return html

@app.route('/systems')
def default():
    html = '<select name="systems" id="systems" onchange="systemChange(this);"><option value="">Select your system</option>'

    html += '<option value="1">2over1</option>'           
    html += '<option value="2">BBO Advanced 1.3</option>'
    html += '<option value="3">SAYC</option>'
    html += '</select> '
    return html

def create_auction(bids, dealer_i, sameforboth):
    auction = [bid.replace('--', "PASS").replace('Db', 'X').replace('Rd', 'XX') for bid in bids]
    if sameforboth:
        auction = ['PAD_START'] * (dealer_i % 2) + auction
    else:
        auction = ['PAD_START'] * dealer_i + auction
    return auction

@app.route('/bid')
def frontend():
    # First we extract our hand
    hand = request.query.get("hand").replace('_','.')
    seat = request.query.get("seat")
    #print(hand)
    # Then vulnerability
    v = request.query.get("vul")
    #print(v)
    vuln = []
    vuln.append('@v' in v)
    vuln.append('@V' in v)
    # And finally the bidding, where we deduct dealer and our position
    dealerInp = request.query.get("dealer")
    dealer = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
    dealer_i = dealer[dealerInp]
    position = dealer[seat]
    ctx = request.query.get("ctx")
    # Split the string into chunks of every second character
    bids = [ctx[i:i+2] for i in range(0, len(ctx), 2)]
    auction = create_auction(bids, dealer_i, models.sameforboth)
    hint_bot = BotBid(vuln, hand, models, sampler, position, dealer_i, verbose)
    bid = hint_bot.bid(auction)
    print(bid.bid)
    player_pool = CardPlayerPool()
    return json.dumps(bid.to_dict())

@app.route('/lead')
def frontend():
    # First we extract our hand
    hand = request.query.get("hand").replace('_','.')
    seat = request.query.get("seat")
    print(hand)
    # Then vulnerability
    v = request.query.get("vul")
    print(v)
    vuln = []
    vuln.append('@v' in v)
    vuln.append('@V' in v)
    # And finally the bidding, where we deduct dealer and our position
    dealerInp = request.query.get("dealer")
    dealer = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
    dealer_i = dealer[dealerInp]
    position = dealer[seat]
    ctx = request.query.get("ctx")
    # Split the string into chunks of every second character
    bids = [ctx[i:i+2] for i in range(0, len(ctx), 2)]
    auction = create_auction(bids, dealer_i, False)
    hint_bot = BotLead(vuln, hand, models, sampler, position, verbose)
    card_resp = hint_bot.find_opening_lead(auction)
    card_symbol = card_resp.card.symbol()
    print(card_symbol)
    result = {
            'card': card_symbol,
        }
    return json.dumps(result)

@app.route('/trick')
def trick():
    result = {
            'status': "OK"
        }
    return json.dumps(result)

@app.route('/updateplay')

def updatePlay():
    result = {
            'status': "OK"
        }
    return json.dumps(result)


@app.route('/play')
def frontend():
    # First we extract our hand
    hand_str = request.query.get("hand").replace('_','.')
    dummy_str = request.query.get("dummy").replace('_','.')
    played = request.query.get("played")
    seat = request.query.get("seat")
    print(hand_str, dummy_str)
    # Then vulnerability
    v = request.query.get("vul")
    # And finally the bidding, where we deduct dealer and our position
    dealerInp = request.query.get("dealer")
    dealer = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
    dealer_i = dealer[dealerInp]
    position_i = dealer[seat]
    vuln = []
    if position_i % 2 == 0:
        vuln.append('@v' in v)
        vuln.append('@V' in v)
    else:
        vuln.append('@V' in v)
        vuln.append('@v' in v)
    ctx = request.query.get("ctx")
    # Split the string into chunks of every second character
    bids = [ctx[i:i+2] for i in range(0, len(ctx), 2)]
    auction = create_auction(bids, dealer_i, False)
    contract = bidding.get_contract(auction)
    print(contract)
    decl_i = bidding.get_decl_i(contract)
    print(decl_i)
    user = request.query.get("user")
    print(player_pool)
    state = player_pool.get_player(user)
    print(state)
    if state == None:
        # This is first time we are playing, so dummy is needed in the request
        state = Player(user)

        is_decl_vuln = [vuln[0], vuln[1], vuln[0], vuln[1]][decl_i]
        state.cardplayer_i = (position_i + 3 - decl_i) % 4
        own_hand_str = hand_str
        dummy_hand_str = '...'

        if state.cardplayer_i != 1:
            dummy_hand_str = dummy_str

        lefty_hand_str = '...'
        if state.cardplayer_i == 0:
            lefty_hand_str = own_hand_str
        
        righty_hand_str = '...'
        if state.cardplayer_i == 2:
            righty_hand_str = own_hand_str
        
        decl_hand_str = '...'
        if state.cardplayer_i == 3:
            decl_hand_str = own_hand_str

        state.card_players = [
                    CardPlayer(models, 0, lefty_hand_str, dummy_hand_str, contract, is_decl_vuln, sampler, False),
                    CardPlayer(models, 1, dummy_str, hand_str, contract, is_decl_vuln, sampler, False),
                    CardPlayer(models, 2, righty_hand_str, dummy_hand_str, contract, is_decl_vuln, sampler, False),
                    CardPlayer(models, 3, hand_str, dummy_hand_str, contract, is_decl_vuln, sampler, False)
            ]
        cards = [played[i:i+2] for i in range(0, len(played), 2)]
        for i in range(len(cards)):
            card52 = deck52.encode_card(cards[i])
            card32 = deck52.card52to32(card52)
            state.current_trick52.append(card52)
            state.current_trick.append(card32)
            for card_player in state.card_players:
                    card_player.set_card_played(trick_i=state.trick_i, leader_i=state.leader_i, i=state.cardplayer_i, card=card32)

            state.card_players[state.cardplayer_i].set_own_card_played52(card52)
            if state.cardplayer_i == 1:
                for i in [0, 2, 3]:
                    state.card_players[i].set_public_card_played52(card52)
            if state.cardplayer_i == 3:
                state.card_players[1].set_public_card_played52(card52)

            # update shown out state
            if card32 // 8 != state.current_trick[0] // 8:  # card is different suit than lead card
                state.shown_out_suits[state.cardplayer_i].add(state.current_trick[0] // 8)

    else:
        print("state reloaded")
        state.cardplayer_i = (position_i + 3 - decl_i) % 4

    print("cardplayer_i", state.cardplayer_i)
    rollout_states, bidding_scores, c_hcp, c_shp, good_quality = sampler.init_rollout_states(state.trick_i, state.cardplayer_i, state.card_players, state.player_cards_played, state.shown_out_suits, state.current_trick, dealer_i, auction, state.card_players[state.cardplayer_i].hand_str, vuln, models, state.card_players[state.cardplayer_i].rng)

    card_resp = state.card_players[state.cardplayer_i].play_card(state.trick_i, state.leader_i, state.current_trick52, rollout_states, bidding_scores, good_quality)
    card_resp.hcp = c_hcp
    card_resp.shape = c_shp

    # Now create the player for playing rest of the hand
    print(state)
    player_pool.set_player(user, state)

    print(player_pool)
    card_symbol = card_resp.card.symbol()
    print(card_symbol)
    result = {
            'card': card_symbol,
        }
    return json.dumps(result)

if __name__ == "__main__":
    run(app, host=host, port=port, server='gevent', log=None)

