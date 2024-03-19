from gevent import monkey
from gevent.pywsgi import WSGIServer

from bots import BotBid, BotLead, CardPlayer
from bidding import bidding
from objects import Card, CardResp
import deck52
from pimc.PIMC import BGADLL
monkey.patch_all()

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import logging

# Set logging level to suppress warnings
logging.getLogger().setLevel(logging.ERROR)
# Just disables the warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# This import is only to help PyInstaller when generating the executables
import tensorflow as tf

import pprint
import argparse
import conf
import numpy as np
from sample import Sample
from urllib.parse import parse_qs, urlparse
from pbn2ben import load

dealer_enum = {'N': 0, 'E': 1, 'S': 2, 'W': 3}

def get_execution_path():
    # Get the directory where the program is started from either PyInstaller executable or the script
    return os.getcwd()

async def play_api(dealer_i, vuln_ns, vuln_ew, hands, models, sampler, contract, strain_i, decl_i, auction, play, position, verbose):
    
    level = int(contract[0])
    is_decl_vuln = [vuln_ns, vuln_ew, vuln_ns, vuln_ew][decl_i]
    cardplayer_i = position  # lefty=0, dummy=1, righty=2, decl=3

    lefty_hand = hands[(decl_i + 1) % 4]
    dummy_hand = hands[(decl_i + 2) % 4]
    righty_hand = hands[(decl_i + 3) % 4]
    decl_hand = hands[decl_i]

    if models.pimc_use and cardplayer_i == 3:
        pimc = BGADLL(models, dummy_hand, decl_hand, contract, is_decl_vuln, verbose)
        if verbose:
            print("PIMC",dummy_hand, decl_hand, contract)
    else:
        pimc = None

    card_players = [
        CardPlayer(models, 0, lefty_hand, dummy_hand, contract, is_decl_vuln, sampler, pimc, verbose),
        CardPlayer(models, 1, dummy_hand, decl_hand, contract, is_decl_vuln, sampler, pimc, verbose),
        CardPlayer(models, 2, righty_hand, dummy_hand, contract, is_decl_vuln, sampler, pimc, verbose),
        CardPlayer(models, 3, decl_hand, dummy_hand, contract, is_decl_vuln, sampler, pimc, verbose)
    ]

    player_cards_played = [[] for _ in range(4)]
    player_cards_played52 = [[] for _ in range(4)]
    shown_out_suits = [set() for _ in range(4)]

    leader_i = 0

    tricks = []
    tricks52 = []
    trick_won_by = []

    opening_lead52 = Card.from_symbol(play[0]).code()
    opening_lead = deck52.card52to32(opening_lead52)

    current_trick = [opening_lead]
    current_trick52 = [opening_lead52]

    card_players[0].hand52[opening_lead52] -= 1
    card_i = 0

    for trick_i in range(13):
        if trick_i != 0:
            print(f"trick {trick_i+1} lead:{leader_i}")

        for player_i in map(lambda x: x % 4, range(leader_i, leader_i + 4)):
            if verbose:
                print('player {}'.format(player_i))
            
            if trick_i == 0 and player_i == 0:
                # To get the state right we ask for the play when using Tf.2X
                if verbose:
                    print('skipping opening lead for ',player_i)
                for i, card_player in enumerate(card_players):
                    card_player.set_real_card_played(opening_lead52, player_i)
                    card_player.set_card_played(trick_i=trick_i, leader_i=leader_i, i=0, card=opening_lead)
                continue

            card_i += 1
            if card_i >= len(play):
                if isinstance(card_players[player_i], CardPlayer):
                    rollout_states, bidding_scores, c_hcp, c_shp, good_quality, probability_of_occurence = sampler.init_rollout_states(trick_i, player_i, card_players, player_cards_played, shown_out_suits, current_trick, dealer_i, auction, card_players[player_i].hand_str, [vuln_ns, vuln_ew], models, card_players[player_i].rng)
                    assert rollout_states[0].shape[0] > 0, "No samples for DDSolver"

                else: 
                    rollout_states = []
                    bidding_scores = []
                    c_hcp = -1
                    c_shp = -1
                    good_quality = None
                    probability_of_occurence = []
                    

                card_resp = None
                while card_resp is None:
                    card_resp = await card_players[player_i].play_card(trick_i, leader_i, current_trick52, rollout_states, bidding_scores, good_quality, probability_of_occurence, shown_out_suits)

                card_resp.hcp = c_hcp
                card_resp.shape = c_shp
                if verbose:
                    pprint.pprint(card_resp.to_dict(), width=200)
                
                return card_resp

            card52 = Card.from_symbol(play[card_i]).code()
            #print(play[card_i], card52, card_i, player_i, cardplayer_i)
            card32 = deck52.card52to32(card52)

            for card_player in card_players:
                card_player.set_real_card_played(card52, player_i)
                card_player.set_card_played(trick_i=trick_i, leader_i=leader_i, i=player_i, card=card32)

            current_trick.append(card32)

            current_trick52.append(card52)

            card_players[player_i].set_own_card_played52(card52)
            if player_i == 1:
                for i in [0, 2, 3]:
                    card_players[i].set_public_card_played52(card52)
            if player_i == 3:
                card_players[1].set_public_card_played52(card52)

            # update shown out state
            if card32 // 8 != current_trick[0] // 8:  # card is different suit than lead card
                shown_out_suits[player_i].add(current_trick[0] // 8)

        # sanity checks after trick completed
        assert len(current_trick) == 4

        # for i in [cardplayer_i] + ([1] if cardplayer_i == 3 else []):
        #     if cardplayer_i == 1:
        #         break
        #     assert np.min(card_player.hand52) == 0, card_player.hand52
        #     assert np.min(card_player.public52) == 0
        #     assert np.sum(card_player.hand52) == 13 - trick_i - 1
        #     assert np.sum(card_player.public52) == 13 - trick_i - 1

        tricks.append(current_trick)
        tricks52.append(current_trick52)

        if models.pimc_use and pimc:
            # Only declarer use PIMC
            if isinstance(card_players[3], CardPlayer):
                card_players[3].pimc.reset_trick()

        # initializing for the next trick
        # initialize hands
        for i, card32 in enumerate(current_trick):
            card_players[(leader_i + i) % 4].x_play[:, trick_i + 1, 0:32] = card_players[(leader_i + i) % 4].x_play[:, trick_i, 0:32]
            card_players[(leader_i + i) % 4].x_play[:, trick_i + 1, 0 + card32] -= 1

        # initialize public hands
        for i in (0, 2, 3):
            card_players[i].x_play[:, trick_i + 1, 32:64] = card_players[1].x_play[:, trick_i + 1, 0:32]
        card_players[1].x_play[:, trick_i + 1, 32:64] = card_players[3].x_play[:, trick_i + 1, 0:32]

        for card_player in card_players:
            # initialize last trick
            for i, card32 in enumerate(current_trick):
                card_player.x_play[:, trick_i + 1, 64 + i * 32 + card32] = 1
                
            # initialize last trick leader
            card_player.x_play[:, trick_i + 1, 288 + leader_i] = 1

            # initialize level
            card_player.x_play[:, trick_i + 1, 292] = level

            # initialize strain
            card_player.x_play[:, trick_i + 1, 293 + strain_i] = 1

        # sanity checks for next trick
        # for i in [cardplayer_i] + ([1] if cardplayer_i == 3 else []):
        #     if cardplayer_i == 1:
        #         break
        #     assert np.min(card_player.x_play[:, trick_i + 1, 0:32]) == 0
        #     assert np.min(card_player.x_play[:, trick_i + 1, 32:64]) == 0
        #     assert np.sum(card_player.x_play[:, trick_i + 1, 0:32], axis=1) == 13 - trick_i - 1
        #     assert np.sum(card_player.x_play[:, trick_i + 1, 32:64], axis=1) == 13 - trick_i - 1

        trick_winner = (leader_i + deck52.get_trick_winner_i(current_trick52, (strain_i - 1) % 5)) % 4
        trick_won_by.append(trick_winner)

        if trick_winner % 2 == 0:
            card_players[0].n_tricks_taken += 1
            card_players[2].n_tricks_taken += 1
        else:
            card_players[1].n_tricks_taken += 1
            card_players[3].n_tricks_taken += 1
            if models.pimc_use and pimc:
                # Only declarer use PIMC
                if isinstance(card_players[3], CardPlayer):
                    card_players[3].pimc.update_trick_needed()

        if verbose:
            print('trick52 {} cards={}. won by {}'.format(trick_i+1, list(map(deck52.decode_card, current_trick52)), trick_winner))

        # update cards shown
        for i, card32 in enumerate(current_trick):
            player_cards_played[(leader_i + i) % 4].append(card32)
        
        for i, card in enumerate(current_trick52):
            player_cards_played52[(leader_i + i) % 4].append(card)

        leader_i = trick_winner
        current_trick = []
        current_trick52 = []


def create_auction(bids, dealer_i):
    auction = [bid.replace('--', "PASS").replace('Db', 'X').replace('Rd', 'XX') for bid in bids]
    auction = ['PAD_START'] * dealer_i + auction
    return auction

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
print(f'http://{host}:{port}/')

app = Flask(__name__)
CORS(app) 

@app.route('/')
def home():
    html = '<h1><a href="/">Play Now</a></h1>\n'
    return html

@app.route('/bid')
def bid():
    try:
        # First we extract our hand
        hand = request.args.get("hand").replace('_','.')
        seat = request.args.get("seat")
        #print(hand)
        # Then vulnerability
        v = request.args.get("vul")
        #print(v)
        vuln = []
        vuln.append('@v' in v)
        vuln.append('@V' in v)
        # And finally the bidding, where we deduct dealer and our position
        dealer = request.args.get("dealer")
        dealer_i = dealer_enum[dealer]
        position = dealer_enum[seat]
        ctx = request.args.get("ctx")
        # Split the string into chunks of every second character
        bids = [ctx[i:i+2] for i in range(0, len(ctx), 2)]
        auction = create_auction(bids, dealer_i)
        hint_bot = BotBid(vuln, hand, models, sampler, position, dealer_i, verbose)
        bid = hint_bot.bid(auction)
        print(bid.bid)
        return json.dumps(bid.to_dict())
    except Exception as e:
        error_message = "An error occurred: {}".format(str(e))
        return jsonify({"error": error_message}), 400  # HTTP status code 500 for internal server error
    
@app.route('/lead')
def lead():
    try:
        # First we extract our hand and seat
        hand = request.args.get("hand").replace('_','.')
        seat = request.args.get("seat")
        # Then vulnerability
        v = request.args.get("vul")
        vuln = []
        vuln.append('@v' in v)
        vuln.append('@V' in v)
        # And finally the dealer and bidding
        dealer = request.args.get("dealer")
        dealer_i = dealer_enum[dealer]
        position = dealer_enum[seat]
        ctx = request.args.get("ctx")
        # Split the string into chunks of every second character
        bids = [ctx[i:i+2] for i in range(0, len(ctx), 2)]
        auction = create_auction(bids, dealer_i)
        hint_bot = BotLead(vuln, hand, models, sampler, position, dealer_i, verbose)
        card_resp = hint_bot.find_opening_lead(auction)
        user = request.args.get("user")
        card_resp.who = user
        print("Leading:", card_resp.card.symbol())
        result = card_resp.to_dict()
        return json.dumps(result)
    except Exception as e:
        error_message = "An error occurred: {}".format(str(e))
        return jsonify({"error": error_message}), 400  # HTTP status code 500 for internal server error


@app.route('/play')
async def frontend():
    #try:
        # First we extract the hands and seat
        hand_str = request.args.get("hand").replace('_','.')
        dummy_str = request.args.get("dummy").replace('_','.')
        if hand_str == dummy_str:
            result = {"message":"Hand and dummy are identical"}
            return json.dumps(result)
        
        played = request.args.get("played")
        seat = request.args.get("seat")
        # Then vulnerability
        v = request.args.get("vul")
        vuln = []
        vuln.append('@v' in v)
        vuln.append('@V' in v)
        # And finally the bidding, where we deduct dealer and our position
        dealer = request.args.get("dealer")
        dealer_i = dealer_enum[dealer]
        position_i = dealer_enum[seat]
        ctx = request.args.get("ctx")
        # Split the string into chunks of every second character
        bids = [ctx[i:i+2] for i in range(0, len(ctx), 2)]
        cards = [played[i:i+2] for i in range(0, len(played), 2)]
        #print(played)
        #print(cards)
        if len(cards) > 51:
            result = {"message": "Game is over, no cards to play"}
            return json.dumps(result)

        # Validate number of cards played according to position
        auction = create_auction(bids, dealer_i)
        contract = bidding.get_contract(auction, dealer_i, models)
        decl_i = bidding.get_decl_i(contract)
        strain_i = bidding.get_strain_i(contract)
        user = request.args.get("user")
        # Hand is following N,E,S,W
        hands = ['...', '...', '...', '...']
        hands[position_i] = hand_str
        if ((decl_i + 2) % 4) == position_i:
            # We are dummy
            hands[decl_i] = dummy_str
        else:        
            hands[(decl_i + 2) % 4] = dummy_str


        # Are we declaring
        if decl_i == position_i:
            cardplayer = 3
        if decl_i == (position_i + 2) % 4:
            cardplayer = 1
        if (decl_i + 1) % 4 == position_i:
            cardplayer = 0
        if (decl_i + 3) % 4 == position_i:
            cardplayer = 2
        #print(cardplayer)
        #print(hands)
        card_resp = await play_api(dealer_i, vuln[0], vuln[1], hands, models, sampler, contract, strain_i, decl_i, auction, cards, cardplayer, verbose)
        print("Playing:", card_resp.card.symbol())
        result = card_resp.to_dict()
        #print(json.dumps(result))
        return json.dumps(result)
    #except Exception as e:
    #    print(e)
    #    error_message = "An error occurred: {}".format(str(e))
    #    return jsonify({"error": error_message}), 400  # HTTP status code 500 for internal server error


if __name__ == "__main__":
    # Run the Flask app with gevent server
    http_server = WSGIServer((host, port), app)
    http_server.serve_forever()