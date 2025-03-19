from gevent import monkey
monkey.patch_all()
import gc
import os
import sys
import platform
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'
# Just disables the warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import logging
import traceback
import util
# Intil fixed in Keras, this is needed to remove a wrong warning
import warnings
warnings.filterwarnings("ignore")

# Set logging level to suppress warnings
logging.getLogger().setLevel(logging.ERROR)

# Configure absl logging to suppress logs
import absl.logging
# Suppress Abseil logs
absl.logging.get_absl_handler().python_handler.stream = open(os.devnull, 'w')
absl.logging.set_verbosity(absl.logging.FATAL)
absl.logging.set_stderrthreshold(absl.logging.FATAL)

import tensorflow as tf
import scoring
import psutil

from gevent.pywsgi import WSGIServer
import datetime 
import time

from bots import BotBid, BotLead, CardPlayer
from bidding import bidding
from objects import Card, CardResp
import deck52
import binary

from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
import json
import logging
from logging.handlers import TimedRotatingFileHandler
from threading import Lock

# Intil fixed in Keras, this is needed to remove a wrong warning
import warnings
warnings.filterwarnings("ignore")

# Set logging level to suppress warnings
logging.getLogger().setLevel(logging.CRITICAL)
# Just disables the warnings
import tensorflow as tf

import pprint
import argparse
import conf
import numpy as np
from sample import Sample
from util import get_play_status, get_singleton, get_possible_cards, calculate_seed
from claim import Claimer

dealer_enum = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
from colorama import Fore, Back, Style, init

init()
def handle_exception(e):
    sys.stderr.write(f"{str(e)}\n")
    traceback_str = traceback.format_exception(type(e), e, e.__traceback__)
    traceback_lines = "".join(traceback_str).splitlines()
    file_traceback = []
    for line in reversed(traceback_lines):
        if line.startswith("  File"):
            file_traceback.append(line.strip()) 
    if file_traceback:
        sys.stderr.write(f"{Fore.RED}")
        sys.stderr.write('\n'.join(file_traceback)+'\n')
        sys.stderr.write(f"{Fore.RESET}")

def get_execution_path():
    # Get the directory where the program is started from either PyInstaller executable or the script
    return os.getcwd()

def play_api(dealer_i, vuln_ns, vuln_ew, hands, models, sampler, contract, strain_i, decl_i, auction, play, cardplayer_i, claim, aceking, verbose):
    
    level = int(contract[0])
    is_decl_vuln = [vuln_ns, vuln_ew, vuln_ns, vuln_ew][decl_i]

    lefty_hand_str = hands[(decl_i + 1) % 4]
    dummy_hand_str = hands[(decl_i + 2) % 4]
    righty_hand_str = hands[(decl_i + 3) % 4]
    decl_hand_str = hands[decl_i]

    pimc = [None, None, None, None]

    # We should only instantiate the PIMC for the position we are playing
    if models.pimc_use_declaring and cardplayer_i == 3: 
        from pimc.PIMC import BGADLL
        declarer = BGADLL(models, dummy_hand_str, decl_hand_str, contract, is_decl_vuln, sampler, verbose)
        pimc[1] = declarer
        pimc[3] = declarer
        if verbose:
            print("PIMC",dummy_hand_str, decl_hand_str, contract)
    else:
        pimc[1] = None
        pimc[3] = None
    if models.pimc_use_defending and (cardplayer_i == 0):
        from pimc.PIMCDef import BGADefDLL
        pimc[0] = BGADefDLL(models, dummy_hand_str, lefty_hand_str, contract, is_decl_vuln, 0, sampler, verbose)
        if verbose:
            print("PIMC",dummy_hand_str, lefty_hand_str, righty_hand_str, contract)
    else:
        pimc[0] = None

    if models.pimc_use_defending and (cardplayer_i == 2):
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
    deck = x = np.ones((52))
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
                deck[opening_lead52] -= 1
                for i, card_player in enumerate(card_players):
                    card_player.set_real_card_played(opening_lead52, player_i)
                    card_player.set_card_played(trick_i=trick_i, leader_i=leader_i, i=0, card=opening_lead)
                continue

            card_i += 1
            if card_i >= len(play):
                if claim:
                    claimer = Claimer(verbose, dds)
                    for i in range(52):
                        if deck[i] != 0: 
                            for j in range(4):
                                if card_players[j].hand52[i] != 0:
                                  deck[i] -= 1  
                    #We need to find the missing cards and distribute between the 2 hidden hands
                    canclaim = claimer.claimapi(
                        strain_i=strain_i,
                        player_i=player_i,
                        hands52=[card_player.hand52 for card_player in card_players],
                        n_samples=1,
                        hidden_cards=deck,
                        current_trick=current_trick52
                    )
                    if (claim <= canclaim):
                        # player_i is relative to declarer
                        claimedbydeclarer = (player_i == 3) or (player_i == 1)
                        if claimedbydeclarer:
                            msg = f"Contract: {contract} Accepted declarers claim of {claim} tricks"
                        else:
                            msg = f"Contract: {contract} Accepted opponents claim of {claim} tricks"
                    else:
                        if claimedbydeclarer:
                            msg = f"Declarer claimed {claim} tricks - rejected {canclaim}"
                        else:
                            msg = f"Opponents claimed {claim} tricks - rejected {canclaim}"
                    return None, player_i, msg

                assert (player_i == cardplayer_i or (player_i == 1 and cardplayer_i == 3)), f"Cardplay order is not correct {play} {player_i} {cardplayer_i} (or another player to play a card)"
                play_status = get_play_status(card_players[player_i].hand52,current_trick52, strain_i)
                if verbose:
                    print("play_status", play_status)

                if play_status == "Forced" and models.autoplaysingleton:
                    card = get_singleton(card_players[player_i].hand52,current_trick52)
                    card_resp = CardResp(
                        card=Card.from_code(card),
                        candidates=[],
                        samples=[],
                        shape=-1,
                        hcp=-1, 
                        quality=None,
                        who="Forced",
                        claim = -1
                    )
                    return card_resp, player_i, play_status
                # if play status = follow 
                # and all out cards are equal value (like JT9)
                # the play lowest if defending and highest if declaring
                # and a possible trick
                # Perhaps we should make it random
                if play_status == "Follow":
                    high, low = get_possible_cards(card_players[player_i].hand52,current_trick52)
                    if high != -1:
                        if player_i == 3: 
                            card = high 
                        else: 
                            card = low
                        card_resp = CardResp(
                            card=Card.from_code(card),
                            candidates=[],
                            samples=[],
                            shape=-1,
                            hcp=-1,
                            quality=None,
                            who="Follow", 
                            claim = -1
                        )                        
                        return card_resp, player_i, play_status
                played_cards = [card for row in player_cards_played52 for card in row] + current_trick52
                print("played cards",played_cards)
                # No obvious play, so we roll out
                rollout_states, bidding_scores, c_hcp, c_shp, quality, probability_of_occurence, lead_scores, play_scores, logical_play_scores, discard_scores, worlds = sampler.init_rollout_states(trick_i, player_i, card_players, played_cards, player_cards_played, shown_out_suits, discards, current_trick, auction, card_players[player_i].hand_str, card_players[player_i].public_hand_str, [vuln_ns, vuln_ew], models, card_players[player_i].get_random_generator())
                assert rollout_states[0].shape[0] > 0, "No samples for DDSolver"
                
                card_players[player_i].check_pimc_constraints(trick_i, rollout_states, quality)

                card_resp =  card_players[player_i].play_card(trick_i, leader_i, current_trick52, tricks52, rollout_states, worlds, bidding_scores, quality, probability_of_occurence, shown_out_suits, play_status, lead_scores, play_scores, logical_play_scores, discard_scores)

                card_resp.hcp = c_hcp
                card_resp.shape = c_shp
                if verbose:
                    print(f"{Fore.LIGHTCYAN_EX}")
                    pprint.pprint(card_resp.to_dict(), width=200)
                    print(f"{Fore.RESET}")
            
                return card_resp, player_i, "Calculated"

            card52 = Card.from_symbol(play[card_i]).code()
            # print(play[card_i], card52, card_i, player_i, cardplayer_i)
            deck[card52] -= 1
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
                discards[player_i].add((trick_i,card32))

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

# Custom function to convert string to boolean
def str_to_bool(value):
    if value.lower() in ['true', '1', 't', 'y', 'yes']:
        return True
    elif value.lower() in ['false', '0', 'f', 'n', 'no']:
        return False
    raise ValueError("Invalid boolean value")


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
parser.add_argument("--verbose", type=str_to_bool, default=False, help="Output samples and other information during play")
parser.add_argument("--port", type=int, default=8085, help="Port for appserver")
parser.add_argument("--record", type=str_to_bool, default=True, help="Recording of responses")
parser.add_argument("--seed", type=int, default=42, help="Seed for random")
parser.add_argument("--matchpoint", type=str_to_bool, default=None, help="Playing match point")

args = parser.parse_args()

configfile = args.config
verbose = args.verbose
port = args.port
record = args.record
matchpoint = args.matchpoint
seed = args.seed

np.set_printoptions(precision=2, suppress=True, linewidth=200)

print(f"{Fore.CYAN}{datetime.datetime.now():%Y-%m-%d %H:%M:%S} gameapi.py - Version 0.8.6.5")
if util.is_pyinstaller_executable():
    print(f"Running inside a PyInstaller-built executable. {platform.python_version()}")
else:
    print(f"Running in a standard Python environment: {platform.python_version()}")

print(f"Python version: {sys.version}{Fore.RESET}")

if sys.platform == 'win32':
    # Print the PythonNet version
    sys.stderr.write(f"PythonNet: {util.get_pythonnet_version()}\n") 
    sys.stderr.write(f"{util.check_dotnet_version()}\n") 

# Try to fetch Keras version or handle older TensorFlow versions
try:
    keras_version = tf.keras.__version__
except AttributeError:
    keras_version = "Not integrated with TensorFlow"
    configfile = configfile.replace("default.conf", "TF1.x/default_tf1x.conf")

# Write to stderr
sys.stderr.write(f"Loading TensorFlow {tf.__version__} - Keras version: {keras_version}\n")

configuration = conf.load(configfile)

try:
    if (configuration["models"]['tf_version'] == "2"):
        from nn.models_tf2 import Models
    else: 
        # Default to version 1. of Tensorflow
        from nn.models import Models
except KeyError:
        # Default to version 1. of Tensorflow
        from nn.models import Models

models = Models.from_conf(configuration, config_path.replace(os.path.sep + "src",""), verbose)
if verbose:
    print("Loading sampler")
sampler = Sample.from_conf(configuration, verbose)

# Improve performance until it is supported
models.claim = False


if sys.platform != 'win32':
    print("Disabling PIMC/BBA/SuitC as platform is not win32")
    models.pimc_use_declaring = False
    models.pimc_use_defending = False
    models.use_bba = False
    models.consult_bba = False
    models.use_bba_rollout = False
    models.use_bba_to_count_aces = False
    models.use_suitc = False
    
print("Config:", configfile)
print("System:", models.name)

if models.use_bba:
    print("Using BBA for bidding")
else:
    print("Model:   ", models.bidder_model.model_path)
    print("Opponent:", models.opponent_model.model_path)

if matchpoint is not None:
    models.matchpoint = matchpoint

if models.matchpoint:
    print("Matchpoint mode on")
else:
    print("Playing IMPS mode")

if models.use_bba or models.use_bba_to_count_aces or models.consult_bba or models.use_bba_rollout:
    from bba.BBA import BBABotBid
    bot = BBABotBid(None, None, None, None, None, None, None, None)
    print(f"BBA enabled. Version {bot.version()}")    

if models.use_suitc:
    from suitc.SuitC import SuitCLib
    suitc = SuitCLib(verbose)
    print(f"SuitC enabled. Version {suitc.version()}")

if models.pimc_use_declaring or models.pimc_use_defending:
    from pimc.PIMC import BGADLL
    pimc = BGADLL(None, None, None, None, None, None, None)
    from pimc.PIMCDef import BGADefDLL
    pimcdef = BGADefDLL(None, None, None, None, None, None, None, None)
    print(f"PIMC enabled. Version {pimc.version()}")
    print(f"PIMCDef enabled. Version {pimcdef.version()}")

from ddsolver import ddsolver
dds = ddsolver.DDSolver()
print(f"DDSolver enabled. Version {dds.version()}")

log_file_path = os.path.join(config_path, 'logs')
if not os.path.exists(log_file_path):
    os.makedirs(log_file_path)
    print(f"Directory '{log_file_path}' created.")

print(f"Setting seed = {seed}")
np.random.seed(seed)
tf.random.set_seed(seed)

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

def replace_x(input_str, rng, dummy, cards, n_cards):
    # Function to replace 'x' in a section with unique digits
    def replace_in_section(suit, section, dummy, cards):
        digits_used = set()  # To keep track of used digits in this section
        for cards in cards:
            if cards[0] == "SHDC"[suit]:
                if cards[1].isdigit():
                    digits_used.add(str(cards[1]))
        for char in dummy:  
            if char.isdigit():
                digits_used.add(char)
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
    dummy_sections = dummy.split('.')
    # Replace 'x' in each section using the corresponding dummy section and pass the index
    replaced_sections = [
        replace_in_section(index, section, dummy_section, cards) 
        for index, (section, dummy_section) in enumerate(zip(sections, dummy_sections))
    ]    
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

# Define allowed hosts
ALLOWED_HOSTS = {'localhost', '127.0.0.1', 'ben.aalborgdata.dk', 'remote.aalborgdata.dk'}
class SilentAbort(HTTPException):
    code = 444  # Non-standard code to terminate without response
    description = "No Response"

def log_request_info():
    # Get the host from the request headers
    host = request.headers.get("Host", "").split(':')[0]  # Ignore port number if present
    # Check if the host is allowed
    if host not in ALLOWED_HOSTS:
        raise SilentAbort()  # Terminates connection silently
    logger.info(f"Request: {request.method} {request.url}")
    if request.method == "POST":
        logger.info(f"Body: {request.get_data()}")

def log_memory_usage():
    # Get system memory info
    virtual_memory = psutil.virtual_memory()
    available_memory = virtual_memory.available / (1024 ** 2)  # Convert bytes to MB
    print(f"Available memory before request: {available_memory:.2f} MB")

@app.before_request
def log_request_and_memory_info():
    log_request_info()  # Call the request logging function
    #log_memory_usage()  # Call the memory usage logging function

@app.after_request
def log_response_info(response):
    if response.status == 444:  # SilentAbort code, ignore status
        return response
    logger.info(f"Response body: {response.status} {response.get_data()}")
    # Get system memory info
    np.empty(0) 
    gc.collect()  # Force garbage collection
    virtual_memory = psutil.virtual_memory()
    available_memory = virtual_memory.available / (1024 ** 2)  # Convert bytes to MB
    print(f"Available memory after request: {available_memory:.2f} MB")
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
        mp = models.matchpoint      
        if request.args.get("tournament"):
            mp = request.args.get("tournament").lower() == "mp"
            models.matchpoint = mp
        details = request.args.get("details")
        if request.args.get("explain"):
            explain = request.args.get("explain").lower() == "true"
        else:
            explain = False
        # First we extract our hand
        hand = request.args.get("hand").replace('_','.').upper()
        if 'X' in hand:
            if '8' in hand or '9' in hand:
                hand = replace_x(hand,get_random_generator(hand), "...", [], models.n_cards_play)
            else:
                hand = replace_x(hand,get_random_generator(hand), "...", [], models.n_cards_bidding)
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
        auction = create_auction(bids, dealer_i)
        if bidding.auction_over(auction):
            result = {"message":"Bidding is over"}
            print(result)
            return json.dumps(result)

        if verbose:
            print("Hand: ",hand)
            print("Vuln: ",vuln)
            print("Dealer: ",dealer)
            print("Seat: ",seat)
            print("Auction: ",auction)
        if models.use_bba:
            from bba.BBA import BBABotBid
            hint_bot = BBABotBid(models.bba_our_cc, models.bba_their_cc, position_i, hand, vuln, dealer_i, mp, verbose)
        else:
            hint_bot = BotBid(vuln, hand, models, sampler, position_i, dealer_i, dds, verbose)
        with model_lock_bid:
            bid = hint_bot.bid(auction)

        print("Bidding: ",bid.bid, "Alert" if bid.alert else "", bid.explanation if bid.explanation else "", f"by {bid.who}" if bid.who else "")
        result = bid.to_dict()
        if not details:
            if "candidates" in result: del result["candidates"]
            if "samples" in result: del result["samples"]
            if "shape" in result: del result["shape"]
            if "hcp" in result: del result["hcp"]

        if record: 
            calculations = {"hand":hand, "vuln":vuln, "dealer":dealer, "seat":seat, "auction":auction, "bid":bid.to_dict()}
            logger.info(f"Calculations bid: {json.dumps(calculations)}")
        print(f'Request took {(time.time() - t_start):0.2f} seconds')       
        return json.dumps(result)
    except Exception as e:
        handle_exception(e)
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
        mp = models.matchpoint     
        if request.args.get("tournament"):
            mp = request.args.get("tournament").lower() == "mp"
            models.matchpoint = mp
        # First we extract our hand and seat
        hand = request.args.get("hand").replace('_','.').upper()
        details = request.args.get("details")
        if 'X' in hand:
            if '8' in hand or '9' in hand:
                hand = replace_x(hand,get_random_generator(hand), "...", [], models.n_cards_play)
            else:
                hand = replace_x(hand,get_random_generator(hand), "...", [], models.n_cards_bidding)
            print(hand)
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
            print(result)
            return json.dumps(result)

        # Find ace and kings
        aceking = {}
        if models.use_bba_to_count_aces:
            from bba.BBA import BBABotBid
            bba_bot = BBABotBid(models.bba_our_cc, models.bba_their_cc, position, hand, vuln, dealer_i, models.matchpoint, verbose)
            aceking = bba_bot.find_aces(auction)
            bba_bot.get_sample(auction)

        hint_bot = BotLead(vuln, hand, models, sampler, position, dealer_i, dds, verbose)
        with model_lock_play:
            card_resp = hint_bot.find_opening_lead(auction, aceking)
        user = request.args.get("user")
        #card_resp.who = user
        print("Leading:", card_resp.card.symbol())
        result = card_resp.to_dict()
        if not details:
            if "candidates" in result: del result["candidates"]
            if "samples" in result: del result["samples"]
            if "shape" in result: del result["shape"]
            if "hcp" in result: del result["hcp"]
        if record: 
            calculations = {"hand":hand, "vuln":vuln, "dealer":dealer, "seat":seat, "auction":auction,  "lead":result}
            logger.info(f"Calculations lead: {json.dumps(calculations)}")
        print(f'Request took {(time.time() - t_start):0.2f} seconds')       
        return json.dumps(result)
    except Exception as e:
        handle_exception(e)
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
        mp = models.matchpoint
        if request.args.get("tournament"):
            mp = request.args.get("tournament").lower() == "mp"
            models.matchpoint = mp
        # First we extract the hands and seat
        hand_str = request.args.get("hand").replace('_','.')
        dummy_str = request.args.get("dummy").replace('_','.')
        played = request.args.get("played")
        details = request.args.get("details")
        cards = [played[i:i+2] for i in range(0, len(played), 2)]
        #print(played)
        #print(cards, len(cards))
        if len(cards) > 51:
            result = {"message": "Game is over, no cards to play"}
            print(result)
            return json.dumps(result)
        if 'X' in hand_str:
            if '8' in hand_str or '9' in hand_str:
                hand_str = replace_x(hand_str,get_random_generator(hand_str), dummy_str, cards, models.n_cards_play)
            else:
                hand_str = replace_x(hand_str,get_random_generator(hand_str), dummy_str, cards, models.n_cards_bidding)
            print(hand_str)
        if hand_str == dummy_str:
            result = {"message":"Hand and dummy are identical"}
            print(result)
            return json.dumps(result)

        if "" == dummy_str:
            result = {"message":"No dummy provided"}
            print(result)
            return json.dumps(result)
        
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
        # Validate number of cards played according to position
        auction = create_auction(bids, dealer_i)
        contract = bidding.get_contract(auction)
        decl_i = bidding.get_decl_i(contract)
        strain_i = bidding.get_strain_i(contract)
        user = request.args.get("user")
        play_pbn_format = request.args.get("format")
        if request.args.get("format"):
            play_pbn_format = request.args.get("format").lower() == "true"
        else:
            play_pbn_format = False

        if play_pbn_format:
            cards = extract_from_pbn(cards, strain_i)

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
        if cardplayer == 1:
            result = {"message": f"Called as dummy or with wrong dealer / seat {contract}"}
            print(result)
            return json.dumps(result)

        # Find ace and kings, when defending
        aceking = {}
        with model_lock_play:
            card_resp, player_i, msg =  play_api(dealer_i, vuln[0], vuln[1], hands, models, sampler, contract, strain_i, decl_i, auction, cards, cardplayer, False, aceking, verbose)
        print("Playing:", card_resp.card.symbol(), msg)
        result = card_resp.to_dict()
        if not details:
            if "candidates" in result: del result["candidates"]
            if "samples" in result: del result["samples"]
            if "shape" in result: del result["shape"]
            if "hcp" in result: del result["hcp"]
        result["player"] = cardplayer
        result["matchpoint"] = mp
        result["MP_or_IMP"] = models.use_real_imp_or_mp
        if record: 
            calculations = {"hand":hand_str, "dummy":dummy_str, "vuln":vuln, "dealer":dealer, "seat":seat, "auction":auction, "play":result}
            logger.info(f"Calculations play: {json.dumps(calculations)}")
        print(f'Request took {(time.time() - t_start):0.2f} seconds')       
        return json.dumps(result)
    except Exception as e:
        print(e)
        handle_exception(e)
        error_message = "An error occurred: {}".format(str(e))
        return jsonify({"error": error_message}), 400  # HTTP status code 500 for internal server error

def get_binary_contract(position, vuln, hand_str, dummy_str, n_cards=32):
    X = np.zeros(2 + 2 * n_cards, dtype=np.float16)

    v_we = vuln[0] if position % 2 == 0 else vuln[1]
    v_them = vuln[1] if position % 2 == 0 else vuln[0]
    vuln = np.array([[v_we, v_them]], dtype=np.float16)
    
    hand = binary.parse_hand_f(n_cards)(hand_str).reshape(n_cards)
    dummy = binary.parse_hand_f(n_cards)(dummy_str).reshape(n_cards)
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
    if record: 
        logger.info(f"cuebidscores: {data}")
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
    turn = data["turn"]
    #{ "bid": "3H", "partnerBidAlert": "4-5!S", "partnerBidAlertArtificial": false, "alert": "3-5!H 1+!C\\nSlam Try\\nForcing one", "artificial": false}
    dealer_i = dealer_enum[dealer]
    position_i = (dealer_i + len(auction_input)) % 4
    auction = ['PAD_START'] * dealer_i + [bid.upper() for bid in auction_input]
    auction = [bid.replace('--', "PASS").replace('Db', 'X').replace('Rd', 'XX').replace("NT","N") for bid in auction]

    vuln_ns = vuln_input == 'NS' or vuln_input == 'ALL'
    vuln_ew = vuln_input == 'EW' or vuln_input == 'ALL'

    vuln = [vuln_ns, vuln_ew]

    if models.use_bba:
        from bba.BBA import BBABotBid
        hint_bot = BBABotBid(models.bba_our_cc, models.bba_their_cc, position_i, hand, vuln, dealer_i, models.matchpoint, verbose)
    else:
        hint_bot = BotBid(vuln, hand, models, sampler, position_i, dealer_i, dds, verbose)
    with model_lock_bid:
        bid = hint_bot.bid(auction)
    result = bid.to_dict()
    explanation = ""
    aler = False
    if explain:
        auction.append(bid.bid)
        if models.use_bba:
            explanation, alert = hint_bot.explain_last_bid(auction)
        else:
            explanation, alert = hint_bot.bbabot.explain_last_bid(auction)
        result["explanation"] = explanation
    result = {"bid": bid.bid.replace("PASS","Pass"), "alert": explanation, "artificial" : alert}
    if record: 
        calculations = {"hand":hand, "vuln":vuln, "dealer":dealer, "turn":turn, "auction":auction, "bid":bid.to_dict()}
        logger.info(f"Calculations cuebid: {json.dumps(calculations)}")

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
    mp = models.matchpoint
    if request.args.get("tournament"):
        mp = request.args.get("tournament").lower() == "mp"
        models.matchpoint = mp
    # And finally we deduct our position
    position_i = dealer_enum[seat]
    dealer = request.args.get("dealer")
    dealer_i = dealer_enum[dealer]
    if verbose:
        print("models.bba_our_cc", models.bba_our_cc, "models.bba_their_cc", models.bba_their_cc)
    bot = BBABotBid(models.bba_our_cc, models.bba_their_cc, position_i, "KJ53.KJ7.AT92.K5", vuln, dealer_i, mp, verbose)
    ctx = request.args.get("ctx")
    # Split the string into chunks of every second character
    bids = [ctx[i:i+2] for i in range(0, len(ctx), 2)]

    auction = create_auction(bids, dealer_i)

    explanation, alert = bot.explain_last_bid(auction)
    
    result = {"explanation": explanation, "Alert": alert} # explaination
    print(f'Request took {(time.time() - t_start):0.2f} seconds')       

    return json.dumps(result)

@app.route('/contract')
def contract():
    try:
        t_start = time.time()
        # First we extract the hands and seat
        hand_str = request.args.get("hand").replace('_','.')
        dummy_str = request.args.get("dummy").replace('_','.')
        if 'X' in hand_str:
            hand_str = replace_x(hand_str,get_random_generator(hand_str), dummy_str, [], models.n_cards_bidding)
        if 'X' in dummy_str:
            dummy_str = replace_x(dummy_str,get_random_generator(dummy_str), hand_str, [], models.n_cards_bidding)
        seat = request.args.get("seat")
        # Then vulnerability
        v = request.args.get("vul")
        vuln = []
        vuln.append('@v' in v)
        vuln.append('@V' in v)
        # And finally we deduct our position
        position_i = dealer_enum[seat]
        X = get_binary_contract(position_i, vuln, hand_str, dummy_str, models.n_cards_bidding)
        result = {}
        with model_lock_bid:
            if verbose:
                print(position_i, vuln, hand_str, dummy_str)
                print(X)
            contract_id = models.contract_model.pred_fun(X)
            contract_id = contract_id.numpy()
            for i in range(len(contract_id[0])):
                if contract_id[0][i] > 0.05:
                    y = np.zeros(5)
                    suit = bidding.ID2BID[i][1]
                    strain_i = 'NSHDC'.index(suit)
                    y[strain_i] = 1
                    Xt = [np.concatenate((X[0], y), axis=0)]
                    tricks = models.trick_model.pred_fun(Xt)
                    tricks = tricks.numpy()
                    for j in range(14):
                        if tricks[0][j] > 0.05:
                            if bidding.ID2BID[i] in result:
                                # Append new data to the existing entry
                                result[bidding.ID2BID[i]]["Tricks"].append(j)
                                result[bidding.ID2BID[i]]["Percentage"].append(round(float(tricks[0][j]), 2))
                            else:
                                # Create a new entry
                                result[bidding.ID2BID[i]] = {
                                    "score": round(float(contract_id[0][i]), 2),
                                    "Tricks": [j],
                                    "Percentage": [round(float(tricks[0][j]), 2)]
                                }                    

            print(f'Request took {(time.time() - t_start):0.2f} seconds')       
        return json.dumps(result)    
    except Exception as e:
        print(e)
        handle_exception(e)
        error_message = "An error occurred: {}".format(str(e))
        return jsonify({"error": error_message}), 400  # HTTP status code 500 for internal server error

@app.route('/claim')
def claim():
    try:
        t_start = time.time()
        claim = request.args.get("tricks")
        # First we extract the hands and seat
        hand_str = request.args.get("hand").replace('_','.')
        dummy_str = request.args.get("dummy").replace('_','.')
        played = request.args.get("played")
        details = request.args.get("details")
        cards = [played[i:i+2] for i in range(0, len(played), 2)]
        #print(played)
        #print(cards, len(cards))
        if len(cards) > 51:
            result = {"message": "Game is over, no claim"}
            print(result)
            return json.dumps(result)
        if hand_str == dummy_str:
            result = {"message":"Hand and dummy are identical"}
            print(result)
            return json.dumps(result)

        if "" == dummy_str:
            result = {"message":"No dummy provided"}
            print(result)
            return json.dumps(result)
        
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
        # Validate number of cards played according to position
        auction = create_auction(bids, dealer_i)
        contract = bidding.get_contract(auction)
        decl_i = bidding.get_decl_i(contract)
        strain_i = bidding.get_strain_i(contract)
        user = request.args.get("user")
        play_pbn_format = request.args.get("format")
        if request.args.get("format"):
            play_pbn_format = request.args.get("format").lower() == "true"
        else:
            play_pbn_format = False

        if play_pbn_format:
            cards = extract_from_pbn(cards, strain_i)

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
        if cardplayer == 1:
            result = {"message": f"Called as dummy or with wrong dealer / seat {contract}"}
            print(result)
            return json.dumps(result)

        #print(hands)
        if not claim:
            claim = 13 - len(cards) // 4
        result = {"tricks": claim}
        # Find ace and kings, when defending
        aceking = {}

        with model_lock_play:
            card_resp, player_i, msg =  play_api(dealer_i, vuln[0], vuln[1], hands, models, sampler, contract, strain_i, decl_i, auction, cards, cardplayer, claim, aceking, verbose)
        result["result"] = msg
        if record: 
            calculations = {"hand":hand_str, "dummy":dummy_str, "vuln":vuln, "dealer":dealer, "seat":seat, "auction":auction, "play":result, "claim":claim}
            logger.info(f"Calculations play: {json.dumps(calculations)}")
        print(f'Request took {(time.time() - t_start):0.2f} seconds')       
        return json.dumps(result)
    except Exception as e:
        print(e)
        handle_exception(e)
        error_message = "An error occurred: {}".format(str(e))
        return jsonify({"error": error_message}), 400 

def extract_from_pbn(cards, strain_i):
    tricks = []
    current_trick = []
            # Use a loop to group every 4 cards as a trick
    for i in range(0, len(cards), 4):
        ct = cards[i:i+4]
        for card in cards[i:i+4]:
            current_trick.append(deck52.encode_card(card))
        if len(ct) == 4:
            tricks.append(current_trick)
            current_trick = []

    if len(tricks) > 0:
        ordered_tricks = [tricks[0]]
        for i in range(1, len(tricks)):
            previous_ordered_trick = ordered_tricks[-1]
            winner_index_in_ordered = deck52.get_trick_winner_i(previous_ordered_trick,  (strain_i - 1) % 5)
                    # Map winner's position back to the original trick's positions
            original_trick = tricks[i - 1]
            winner_card = previous_ordered_trick[winner_index_in_ordered]
            winner_index_in_original = original_trick.index(winner_card)
                    
                    # Rotate the next trick to start with the winner's position in the original order
            next_trick = tricks[i]
            reordered_trick = next_trick[winner_index_in_original:] + next_trick[:winner_index_in_original]
            ordered_tricks.append(reordered_trick)

        pbn_cards = []
        for trick in ordered_tricks:
            for card in trick:
                card = deck52.decode_card(card)
                pbn_cards.append(card)
                
        cards = pbn_cards
        for card in current_trick:
            cards.append(deck52.decode_card(card))
    return cards # HTTP status code 500 for internal server error

if __name__ == "__main__":
    print(Back.BLACK)
    try:
        # Run the Flask app with gevent server
        http_server = WSGIServer((host, port), app)
        http_server.spawn = 4 #Create 4 Workers
        http_server.connection_timeout = 120  # Set timeout to 120 seconds
        http_server.serve_forever()
    finally:
        print(Style.RESET_ALL)        