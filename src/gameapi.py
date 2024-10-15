import sys
import traceback
from gevent import monkey
monkey.patch_all()
from gevent.pywsgi import WSGIServer
import datetime 
import time

from bots import BotBid, BotLead, CardPlayer
from bidding import bidding
from objects import Card, CardResp
import deck52
import binary

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import logging
from logging.handlers import TimedRotatingFileHandler
import uuid
import shelve
from threading import Lock

# Set logging level to suppress warnings
logging.getLogger().setLevel(logging.ERROR)
# Just disables the warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

# Configure absl logging to suppress logs
import absl.logging
# Suppress Abseil logs
absl.logging.get_absl_handler().python_handler.stream = open(os.devnull, 'w')
absl.logging.set_verbosity(absl.logging.FATAL)
absl.logging.set_stderrthreshold(absl.logging.FATAL)
# This import is only to help PyInstaller when generating the executables
import tensorflow as tf

import pprint
import argparse
import conf
import numpy as np
from sample import Sample
from util import get_play_status, get_singleton, get_possible_cards, calculate_seed

dealer_enum = {'N': 0, 'E': 1, 'S': 2, 'W': 3}

def get_execution_path():
    # Get the directory where the program is started from either PyInstaller executable or the script
    return os.getcwd()

def play_api(dealer_i, vuln_ns, vuln_ew, hands, models, sampler, contract, strain_i, decl_i, auction, play, position, verbose):
    
    level = int(contract[0])
    is_decl_vuln = [vuln_ns, vuln_ew, vuln_ns, vuln_ew][decl_i]
    cardplayer_i = position  # lefty=0, dummy=1, righty=2, decl=3

    lefty_hand_str = hands[(decl_i + 1) % 4]
    dummy_hand_str = hands[(decl_i + 2) % 4]
    righty_hand_str = hands[(decl_i + 3) % 4]
    decl_hand_str = hands[decl_i]

    pimc = [None, None, None, None]

    # We should only instantiate the PIMC for the position we are playing
    if models.pimc_use_declaring and position == 3: 
        from pimc.PIMC import BGADLL
        declarer = BGADLL(models, dummy_hand_str, decl_hand_str, contract, is_decl_vuln, sampler, verbose)
        pimc[1] = declarer
        pimc[3] = declarer
        if verbose:
            print("PIMC",dummy_hand_str, decl_hand_str, contract)
    else:
        pimc[1] = None
        pimc[3] = None
    if models.pimc_use_defending and (position == 0):
        from pimc.PIMCDef import BGADefDLL
        pimc[0] = BGADefDLL(models, dummy_hand_str, lefty_hand_str, contract, is_decl_vuln, 0, sampler, verbose)
        if verbose:
            print("PIMC",dummy_hand_str, lefty_hand_str, righty_hand_str, contract)
    else:
        pimc[0] = None

    if models.pimc_use_defending and (position == 2):
        from pimc.PIMCDef import BGADefDLL
        pimc[2] = BGADefDLL(models, dummy_hand_str, righty_hand_str, contract, is_decl_vuln, 2, sampler, verbose)
        if verbose:
            print("PIMC",dummy_hand_str, lefty_hand_str, righty_hand_str, contract)
    else:
        pimc[2] = None

    card_players = [
        CardPlayer(models, 0, lefty_hand_str, dummy_hand_str, contract, is_decl_vuln, sampler, pimc[0], dds, verbose),
        CardPlayer(models, 1, dummy_hand_str, decl_hand_str, contract, is_decl_vuln, sampler, pimc[1], dds, verbose),
        CardPlayer(models, 2, righty_hand_str, dummy_hand_str, contract, is_decl_vuln, sampler, pimc[2], dds, verbose),
        CardPlayer(models, 3, decl_hand_str, dummy_hand_str, contract, is_decl_vuln, sampler, pimc[3], dds, verbose)
    ]

    player_cards_played = [[] for _ in range(4)]
    player_cards_played52 = [[] for _ in range(4)]
    shown_out_suits = [set() for _ in range(4)]
    discards = [set() for _ in range(4)]

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
        if trick_i != 0 and verbose:
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
                assert (player_i == position or (player_i == 1 and position == 3)), f"Cardplay order is not correct {play} {player_i} {position}"
                play_status = get_play_status(card_players[player_i].hand52,current_trick52)
                if verbose:
                    print("play_status", play_status)

                if play_status == "Forced":
                    card = get_singleton(card_players[player_i].hand52,current_trick52)
                    card_resp = CardResp(
                        card=Card.from_code(card),
                        candidates=[],
                        samples=[],
                        shape=-1,
                        hcp=-1, 
                        quality=None,
                        who="Forced"
                    )
                    return card_resp, player_i
                # if play status = follow 
                # and all out cards are equal value (like JT9)
                # the play lowest if defending and highest if declaring
                if play_status == "Follow":
                    result = get_possible_cards(card_players[player_i].hand52,current_trick52)
                    if result[0] != -1:
                        card = result[0] if player_i == 3 else result[1]
                        card_resp = CardResp(
                            card=Card.from_code(card),
                            candidates=[],
                            samples=[],
                            shape=-1,
                            hcp=-1,
                            quality=None,
                            who="Follow"
                        )                        
                        return card_resp, player_i

                # No obvious play, so we roll out
                rollout_states, bidding_scores, c_hcp, c_shp, quality, probability_of_occurence = sampler.init_rollout_states(trick_i, player_i, card_players, player_cards_played, shown_out_suits, current_trick, dealer_i, auction, card_players[player_i].hand_str, card_players[player_i].public_hand_str, [vuln_ns, vuln_ew], models, card_players[player_i].get_random_generator())
                assert rollout_states[0].shape[0] > 0, "No samples for DDSolver"
                
                card_players[player_i].check_pimc_constraints(trick_i, rollout_states, quality)

                card_resp =  card_players[player_i].play_card(trick_i, leader_i, current_trick52, tricks52, rollout_states, bidding_scores, quality, probability_of_occurence, shown_out_suits, play_status)

                card_resp.hcp = c_hcp
                card_resp.shape = c_shp
                if verbose:
                    pprint.pprint(card_resp.to_dict(), width=200)
            
                return card_resp, player_i

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
                discards[player_i].add(card32)

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

        if models.pimc_use_declaring or models.pimc_use_defending:
            for card_player in card_players:
                if isinstance(card_player, CardPlayer) and card_player.pimc:
                    card_player.pimc.reset_trick()

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
            if models.pimc_use_defending:
                if isinstance(card_players[0], CardPlayer) and card_players[0].pimc:
                    card_players[0].pimc.update_trick_needed()
                if isinstance(card_players[2], CardPlayer) and card_players[2].pimc:
                    card_players[2].pimc.update_trick_needed()
        else:
            card_players[1].n_tricks_taken += 1
            card_players[3].n_tricks_taken += 1
            if models.pimc_use_declaring:
                if isinstance(card_players[3], CardPlayer) and card_players[3].pimc :
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
    
parser = argparse.ArgumentParser(description="Game API")
parser.add_argument("--host", default="localhost", help="Hostname for appserver")
parser.add_argument("--config", default=f"{config_path}/config/default_api.conf", help="Filename for configuration")
parser.add_argument("--verbose", type=bool, default=False, help="Output samples and other information during play")
parser.add_argument("--port", type=int, default=8085, help="Port for appserver")
parser.add_argument("--record", type=bool, default=True, help="Recording of responses")

args = parser.parse_args()

configfile = args.config
verbose = args.verbose
port = args.port
record = args.record

np.set_printoptions(precision=2, suppress=True, linewidth=240)

configuration = conf.load(configfile)

try:
    if (configuration["models"]['tf_version'] == "2"):
        sys.stderr.write("Loading tensorflow 2.X\n")
        from nn.models_tf2 import Models
    else: 
        # Default to version 1. of Tensorflow
        from nn.models import Models
except KeyError:
        # Default to version 1. of Tensorflow
        from nn.models import Models

models = Models.from_conf(configuration, config_path.replace(os.path.sep + "src",""))
sampler = Sample.from_conf(configuration, verbose)

# Improve performance until it is supported
models.claim = False

import platform
if sys.platform != 'win32':
    print("Disabling PIMC/BBA/SuitC as platform is not win32")
    models.pimc_use_declaring = False
    models.pimc_use_defending = False
    models.use_bba = False
    models.use_suitc = False
 
from ddsolver import ddsolver
dds = ddsolver.DDSolver() 
log_file_path = os.path.join(config_path, 'logs')
if not os.path.exists(log_file_path):
    os.makedirs(log_file_path)
    print(f"Directory '{log_file_path}' created.")

print("Config:", configfile)
print("System:", models.name)
if models.use_bba:
    print("Using BBA for bidding")
else:
    print("Model:", models.bidder_model.model_path)
    print("Opponent:", models.opponent_model.model_path)
if models.matchpoint:
    print("Matchpoint mode on")
else:
    print("Playing IMPS mode")


host = args.host
print(f'http://{host}:{port}/')

app = Flask(__name__)
CORS(app) 

# Initialize the lock
model_lock_bid = Lock()
model_lock_play = Lock()
# Set up logging
class PrefixedTimedRotatingFileHandler(TimedRotatingFileHandler):
    def __init__(self, prefix, when='midnight', interval=1, backupCount=0):
        self.prefix = prefix
        filename = self.get_filename()
        super().__init__(filename, when, interval, backupCount)

    def get_filename(self):
        date_str = datetime.datetime.now().strftime('%Y-%m-%d')
        return f"{self.prefix}-{host}-{port}-{date_str}.log"

    def doRollover(self):
        self.stream.close()
        self.baseFilename = self.get_filename()
        self.mode = 'a'
        self.stream = self._open()

def get_random_generator(hand):
    hash_integer  = calculate_seed(hand)         
    return np.random.default_rng(hash_integer)

def replace_x(input_str, rng, n_cards):
    # Function to replace 'x' in a section with unique digits
    def replace_in_section(section):
        digits_used = set()  # To keep track of used digits in this section
        result = []
        
        for char in section:
            if char == 'X':
                # Generate a unique digit not already used in this section
                digit = None
                while digit is None or digit in digits_used:
                    digit = str(rng.integers(2, 15 - (n_cards // 4)))  # Random digit as string
                digits_used.add(digit)  # Mark the digit as used
                result.append(digit)
            else:
                result.append(char)
        
        return ''.join(result)

    # Split the input into sections by '.'
    sections = input_str.split('.')

    # Replace 'x' in each section with unique digits
    replaced_sections = [replace_in_section(section) for section in sections]

    # Join the sections back with '.'
    return '.'.join(replaced_sections)

# Set up logging
log_handler = PrefixedTimedRotatingFileHandler(prefix='logs/gameapi', when="midnight", interval=1)
log_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
log_handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.addHandler(log_handler)
logger.setLevel(logging.INFO)

@app.before_request
def log_request_info():
    logger.info(f"Request: {request.method} {request.url}")
    #logger.info(f"Headers: {request.headers}")
    if request.method == "POST":
        logger.info(f"Body: {request.get_data()}")


@app.after_request
def log_response_info(response):
    #logger.info(f"Response status: {response.status}")
    #logger.info(f"Response headers: {response.headers}")
    logger.info(f"Response body: {response.status} {response.get_data()}")
    return response


@app.route('/')
def home():
    html = '<h1><a href="/">Play Now</a></h1>\n'
    return html

@app.route('/bid')
def bid():
    try:
        t_start = time.time()
        if request.args.get("dealno"):
            dealno = request.args.get("dealno")
            dealno = "{}-{}".format(dealno, datetime.datetime.now().strftime("%Y-%m-%d"))    
        else:
            dealno = "-{}".format(datetime.datetime.now().strftime("%Y-%m-%d"))        
        if request.args.get("tournament"):
            matchpoint = request.args.get("tournament").lower() == "mp"
            models.matchpoint = matchpoint
        if request.args.get("explain"):
            explain = request.args.get("explain").lower() == "true"
        else:
            explain = False
        # First we extract our hand
        hand = request.args.get("hand").replace('_','.')
        if 'X' in hand:
            hand = replace_x(hand,get_random_generator(hand), models.n_cards_bidding)
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
        position_i = dealer_enum[seat]
        ctx = request.args.get("ctx")
        # Split the string into chunks of every second character
        bids = [ctx[i:i+2] for i in range(0, len(ctx), 2)]
        auction = create_auction(bids, dealer_i)
        if bidding.auction_over(auction):
            result = {"message":"Bidding is over"}
            return json.dumps(result)

        if verbose:
            print("Hand: ",hand)
            print("Vuln: ",vuln)
            print("Dealer: ",dealer)
            print("Seat: ",seat)
            print("Auction: ",auction)
        if models.use_bba:
            from bba.BBA import BBABotBid
            hint_bot = BBABotBid(models.bba_ns, models.bba_ew, position_i, "KJ53.KJ7.AT92.K5", vuln, dealer_i, models.matchpoint)
        else:
            hint_bot = BotBid(vuln, hand, models, sampler, position_i, dealer_i, dds, verbose)
        with model_lock_bid:
            bid = hint_bot.bid(auction)
        print("Bidding: ",bid.bid, "Alert" if bid.alert else "")
        result = bid.to_dict()
        if explain:
            from bba.BBA import BBABotBid
            print("models.bba_ns", models.bba_ns, "models.bba_ew", models.bba_ew)
            bot = BBABotBid(models.bba_ns, models.bba_ew, position_i, "KJ53.KJ7.AT92.K5", vuln, dealer_i, models.matchpoint)
            auction.append(bid.bid)
            result["explanation"] = bot.explain(auction)
            print("explanation: ",result["explanation"])

        if record: 
            calculations = {"hand":hand, "vuln":vuln, "dealer":dealer, "seat":seat, "auction":auction, "bid":bid.to_dict()}
            logger.info(f"Calulations bid: {calculations}")
        print(f'Request took {(time.time() - t_start):0.2f} seconds')       
        return json.dumps(result)
    except Exception as e:
        print(e)
        traceback_str = traceback.format_exception(type(e), e, e.__traceback__)
        traceback_lines = "".join(traceback_str).splitlines()
        file_traceback = None
        for line in reversed(traceback_lines):
            if line.startswith("  File"):
                file_traceback = line
                break
        if file_traceback:
            print(file_traceback)  # This will print the last section starting with "File"
        error_message = "An error occurred: {}".format(str(e))
        return jsonify({"error": error_message}), 400  # HTTP status code 500 for internal server error    
    
@app.route('/lead')
def lead():
    try:
        t_start = time.time()
        if request.args.get("dealno"):
            dealno = request.args.get("dealno")
            dealno = "{}-{}".format(dealno, datetime.datetime.now().strftime("%Y-%m-%d"))    
        else:
            dealno = "-{}".format(datetime.datetime.now().strftime("%Y-%m-%d"))        
        if request.args.get("tournament"):
            matchpoint = request.args.get("tournament").lower() == "mp"
            models.matchpoint = matchpoint
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
        contract = bidding.get_contract(auction)
        decl_i = bidding.get_decl_i(contract)
        if "NESW"[(decl_i + 1) % 4] != seat:
            result = {"message":"Not this player to lead"}
            return json.dumps(result)

        hint_bot = BotLead(vuln, hand, models, sampler, position, dealer_i, dds, verbose)
        with model_lock_play:
            card_resp = hint_bot.find_opening_lead(auction)
        user = request.args.get("user")
        #card_resp.who = user
        print("Leading:", card_resp.card.symbol())
        result = card_resp.to_dict()
        if record: 
            calculations = {"hand":hand, "vuln":vuln, "dealer":dealer, "seat":seat, "auction":auction,  "lead":result}
            logger.info(f"Calulations lead: {calculations}")
        print(f'Request took {(time.time() - t_start):0.2f} seconds')       
        return json.dumps(result)
    except Exception as e:
        print(e)
        traceback_str = traceback.format_exception(type(e), e, e.__traceback__)
        traceback_lines = "".join(traceback_str).splitlines()
        file_traceback = None
        for line in reversed(traceback_lines):
            if line.startswith("  File"):
                file_traceback = line
                break
        if file_traceback:
            print(file_traceback)  # This will print the last section starting with "File"
        error_message = "An error occurred: {}".format(str(e))
        return jsonify({"error": error_message}), 400  # HTTP status code 500 for internal server error


@app.route('/play')
def play():
    try:
        t_start = time.time()
        if request.args.get("dealno"):
            dealno = request.args.get("dealno")
            dealno = "{}-{}".format(dealno, datetime.datetime.now().strftime("%Y-%m-%d"))    
        else:
            dealno = "-{}".format(datetime.datetime.now().strftime("%Y-%m-%d"))        
        if request.args.get("tournament"):
            matchpoint = request.args.get("tournament").lower() == "mp"
            models.matchpoint = matchpoint
        # First we extract the hands and seat
        hand_str = request.args.get("hand").replace('_','.')
        dummy_str = request.args.get("dummy").replace('_','.')
        if hand_str == dummy_str:
            result = {"message":"Hand and dummy are identical"}
            return json.dumps(result)

        if "" == dummy_str:
            result = {"message":"No dummy provided"}
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
        #print(cards, len(cards))
        if len(cards) > 51:
            result = {"message": "Game is over, no cards to play"}
            return json.dumps(result)

        # Validate number of cards played according to position
        auction = create_auction(bids, dealer_i)
        contract = bidding.get_contract(auction)
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
        #print("cardplayer:",cardplayer)
        #print(hands)
        with model_lock_play:
            card_resp, player_i =  play_api(dealer_i, vuln[0], vuln[1], hands, models, sampler, contract, strain_i, decl_i, auction, cards, cardplayer, verbose)
        print("Playing:", card_resp.card.symbol())
        result = card_resp.to_dict()
        result["player"] = player_i
        result["matchpoint"] = matchpoint
        result["MP_or_IMP"] = models.use_real_imp_or_mp
        if record: 
            calculations = {"hand":hand_str, "dummy":dummy_str, "vuln":vuln, "dealer":dealer, "seat":seat, "auction":auction, "play":result}
            logger.info(f"Calulations play: {calculations}")
        print(f'Request took {(time.time() - t_start):0.2f} seconds')       
        return json.dumps(result)
    except Exception as e:
        print(e)
        traceback_str = traceback.format_exception(type(e), e, e.__traceback__)
        traceback_lines = "".join(traceback_str).splitlines()
        file_traceback = None
        for line in reversed(traceback_lines):
            if line.startswith("  File"):
                file_traceback = line
                break
        if file_traceback:
            print(file_traceback)  # This will print the last section starting with "File"
        error_message = "An error occurred: {}".format(str(e))
        return jsonify({"error": error_message}), 400  # HTTP status code 500 for internal server error

def get_binary_contract(position, vuln, hand_str, dummy_str):
    X = np.zeros(2 + 2 * 32, dtype=np.float16)

    v_we = vuln[0] if position % 2 == 0 else vuln[1]
    v_them = vuln[1] if position % 2 == 0 else vuln[0]
    vuln = np.array([[v_we, v_them]], dtype=np.float16)
    
    hand = binary.parse_hand_f(32)(hand_str).reshape(32)
    dummy = binary.parse_hand_f(32)(dummy_str).reshape(32)
    ftrs = np.concatenate((
        vuln,
        [hand],
        [dummy],
    ), axis=1)
    X = ftrs
    return X


@app.route('/cuebidscores', methods=['POST'])
def cuebidscores():
    t_start = time.time()
    data = request.get_json()
    # Create a filename with the current date
    date_str = time.strftime("%Y-%m-%d")  # Format: YYYY-MM-DD
    filename = f'logs/cuebidscores_{date_str}.log'
    
    # Ensure the logs directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Write or append data to the file
    with open(filename, 'a') as file:
        file.write(json.dumps(data) + '\n')
    
    # log request to log file
    print(data)
    result = {}
    print(f'Request took {(time.time() - t_start):0.2f} seconds')       
    return json.dumps(result),200


@app.route('/cuebid', methods=['POST'])
def cuebid():
    t_start = time.time()
    # Override defaults as cuebids must finish 8 boards within 9 mins
    sampler.no_samples_when_no_search = True
    sampler.sample_boards_for_auction = 5000
    data = request.get_json()
    # log request to log file

    if not data:
        return jsonify({"error": "Invalid or missing JSON"}), 400
    #{"dealer":"N","auction":["1H","Pass"],"vuln":"EW","hand":"95.T982.A5.AKJ85","sysNS":"GAVIN_ADVANCED","sysEW":"DEFAULT","conventions":[]}}}
    dealer = data["dealer"]
    auction_input= data["auction"]
    vuln_input = data["vuln"]
    hand = data["hand"]
    #{ "bid": "3H", "partnerBidAlert": "4-5!S", "partnerBidAlertArtificial": false, "alert": "3-5!H 1+!C\\nSlam Try\\nForcing one", "artificial": false}
    dealer_i = dealer_enum[dealer]
    position_i = (dealer_i + len(auction_input)) % 4
    auction = ['PAD_START'] * dealer_i + [bid.upper() for bid in auction_input]
    auction = [bid.replace('--', "PASS").replace('Db', 'X').replace('Rd', 'XX').replace("NT","N") for bid in auction]

    print("Auction:",auction)

    vuln_ns = vuln_input == 'NS' or vuln_input == 'ALL'
    vuln_ew = vuln_input == 'EW' or vuln_input == 'ALL'

    vuln = [vuln_ns, vuln_ew]

    if models.use_bba:
        from bba.BBA import BBABotBid
        hint_bot = BBABotBid(models.bba_ns, models.bba_ew, position_i, "KJ53.KJ7.AT92.K5", vuln, dealer_i, models.matchpoint)
    else:
        hint_bot = BotBid(vuln, hand, models, sampler, position_i, dealer_i, dds, verbose)
    with model_lock_bid:
        bid = hint_bot.bid(auction)
    print("Bidding: ",bid.bid)
    result = bid.to_dict()
    if explain:
        from bba.BBA import BBABotBid
        print("models.bba_ns", models.bba_ns, "models.bba_ew", models.bba_ew)
        bot = BBABotBid(models.bba_ns, models.bba_ew, position_i, "KJ53.KJ7.AT92.K5", vuln, dealer_i, models.matchpoint)
        auction.append(bid.bid)
        result["explanation"] = bot.explain(auction)
        print("explanation: ",result["explanation"])
    result = {"bid": bid.bid.replace("PASS","Pass"), "alert": bot.explain(auction), "artificial" : bid.alert}
    print(f'Request took {(time.time() - t_start):0.2f} seconds')       
    return json.dumps(result),200

@app.route('/explain')
def explain():
    t_start = time.time()
    from bba.BBA import BBABotBid
    # First we extract the hands and seat
    seat = request.args.get("seat")
    # Then vulnerability
    v = request.args.get("vul")
    vuln = []
    vuln.append('@v' in v)
    vuln.append('@V' in v)
    # And finally we deduct our position
    position_i = dealer_enum[seat]
    dealer = request.args.get("dealer")
    dealer_i = dealer_enum[dealer]
    if verbose:
        print("models.bba_ns", models.bba_ns, "models.bba_ew", models.bba_ew)
    bot = BBABotBid(models.bba_ns, models.bba_ew, position_i, "KJ53.KJ7.AT92.K5", vuln, dealer_i, models.matchpoint)
    ctx = request.args.get("ctx")
    # Split the string into chunks of every second character
    bids = [ctx[i:i+2] for i in range(0, len(ctx), 2)]

    auction = create_auction(bids, dealer_i)

    explanation = bot.explain(auction)
    
    result = {"explanation": explanation} # explaination
    print(f'Request took {(time.time() - t_start):0.2f} seconds')       

    return json.dumps(result)

@app.route('/contract')
def contract():
    t_start = time.time()
    # First we extract the hands and seat
    hand_str = request.args.get("hand").replace('_','.')
    if 'X' in hand_str:
        hand_str = replace_x(hand_str,get_random_generator(hand_str), models.n_cards_bidding)
    dummy_str = request.args.get("dummy").replace('_','.')
    if 'X' in dummy_str:
        dummy_str = replace_x(dummy_str,get_random_generator(dummy_str), models.n_cards_bidding)
    seat = request.args.get("seat")
    # Then vulnerability
    v = request.args.get("vul")
    vuln = []
    vuln.append('@v' in v)
    vuln.append('@V' in v)
    # And finally we deduct our position
    position_i = dealer_enum[seat]
    X = get_binary_contract(position_i, vuln, hand_str, dummy_str)
    with model_lock_bid:
        contract_id, doubled, tricks = models.contract_model.model[0](X)
        contract = bidding.ID2BID[contract_id] + ("X" if doubled else "") 
        result = {"contract": contract,
                "tricks": tricks}
        print(result)
        # New call to get top 3 tricks
        top_k_indices_tricks, top_k_probs_tricks = models.contract_model.model[1](X)
        print(top_k_indices_tricks, top_k_probs_tricks)

        # New call to get top 3 contracts
        top_k_indices_oh, top_k_probs_oh = models.contract_model.model[2](X, k=3)
        print(top_k_indices_oh, top_k_probs_oh)
        print(f'Request took {(time.time() - t_start):0.2f} seconds')       
    return json.dumps(result)    


if __name__ == "__main__":
    # Run the Flask app with gevent server
    http_server = WSGIServer((host, port), app)
    http_server.spawn = 4 #Create 4 Workers
    http_server.connection_timeout = 120  # Set timeout to 120 seconds
    http_server.serve_forever()