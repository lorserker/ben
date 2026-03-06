import faulthandler
faulthandler.enable()
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
import psutil

from gevent.pywsgi import WSGIServer
import datetime 
import time

from botbidder import BotBid
from botopeninglead import BotLead
from botcardplayer import CardPlayer
from bidding import bidding
from objects import Card, CardResp
import deck52
import binary

# Import PIMC exceptions for fallback handling
try:
    from pimc.PIMC import PIMCNoPlayoutError as PIMCNoPlayoutErrorDeclarer
except ImportError:
    PIMCNoPlayoutErrorDeclarer = None
try:
    from pimc.PIMCDef import PIMCNoPlayoutError as PIMCNoPlayoutErrorDefender
except ImportError:
    PIMCNoPlayoutErrorDefender = None

# Import ACE classes for alternative play engine
try:
    from ace.ACE import ACEDLL
except ImportError:
    ACEDLL = None
try:
    from ace.ACEDef import ACEDefDLL
except ImportError:
    ACEDefDLL = None

# Import ACE-MCTS classes
try:
    from ace.ACEMCTS import ACEMCTSDLL
except ImportError:
    ACEMCTSDLL = None
try:
    from ace.ACEMCTSDef import ACEMCTSDefDLL
except ImportError:
    ACEMCTSDefDLL = None

from flask import Flask, Response, request, jsonify, abort
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
import json
import logging
from logging.handlers import TimedRotatingFileHandler
from threading import Lock
from nn.timing import ModelTimer

# Intil fixed in Keras, this is needed to remove a wrong warning
import warnings
warnings.filterwarnings("ignore")

# Set logging level to suppress warnings
logging.getLogger().setLevel(logging.CRITICAL)
# Just disables the warnings
import tensorflow as tf
from nn.opponents import Opponents

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address # Default key function (limits by IP)

import pprint
import argparse
import conf
import numpy as np
from sample import Sample
from util import get_play_status, get_singleton, get_possible_cards, calculate_seed
from claim import Claimer

dealer_enum = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
from colorama import Fore, Back, Style, init

version = '0.8.7.6'
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

def play_api(dealer_i, vuln_ns, vuln_ew, hands, models, sampler, contract, strain_i, decl_i, auction, play, cardplayer_i, claim, features, verbose):
    
    level = int(contract[0])
    is_decl_vuln = [vuln_ns, vuln_ew, vuln_ns, vuln_ew][decl_i]

    lefty_hand_str = hands[(decl_i + 1) % 4]
    dummy_hand_str = hands[(decl_i + 2) % 4]
    righty_hand_str = hands[(decl_i + 3) % 4]
    decl_hand_str = hands[decl_i]

    pimc = [None, None, None, None]

    # Check if engines are enabled (ACE-MCTS > ACE > PIMC priority)
    ace_mcts_use_declaring = getattr(models, 'ace_mcts_use_declaring', False)
    ace_mcts_use_defending = getattr(models, 'ace_mcts_use_defending', False)
    ace_use_declaring = getattr(models, 'ace_use_declaring', False)
    ace_use_defending = getattr(models, 'ace_use_defending', False)

    # We should only instantiate the play engine for the position we are playing
    if ace_mcts_use_declaring and cardplayer_i == 3 and ACEMCTSDLL is not None:
        declarer = ACEMCTSDLL(models, dummy_hand_str, decl_hand_str, contract, is_decl_vuln, sampler, verbose)
        pimc[1] = declarer
        pimc[3] = declarer
        if verbose:
            print("ACE-MCTS", dummy_hand_str, decl_hand_str, contract)
    elif ace_use_declaring and cardplayer_i == 3 and ACEDLL is not None:
        declarer = ACEDLL(models, dummy_hand_str, decl_hand_str, contract, is_decl_vuln, sampler, verbose)
        pimc[1] = declarer
        pimc[3] = declarer
        if verbose:
            print("ACE", dummy_hand_str, decl_hand_str, contract)
    elif models.pimc_use_declaring and cardplayer_i == 3:
        from pimc.PIMC import BGADLL
        declarer = BGADLL(models, dummy_hand_str, decl_hand_str, contract, is_decl_vuln, sampler, verbose)
        pimc[1] = declarer
        pimc[3] = declarer
        if verbose:
            print("PIMC", dummy_hand_str, decl_hand_str, contract)
    else:
        pimc[1] = None
        pimc[3] = None

    if ace_mcts_use_defending and (cardplayer_i == 0) and ACEMCTSDefDLL is not None:
        pimc[0] = ACEMCTSDefDLL(models, dummy_hand_str, lefty_hand_str, contract, is_decl_vuln, 0, sampler, verbose)
        if verbose:
            print("ACE-MCTS", dummy_hand_str, lefty_hand_str, righty_hand_str, contract)
    elif ace_use_defending and (cardplayer_i == 0) and ACEDefDLL is not None:
        pimc[0] = ACEDefDLL(models, dummy_hand_str, lefty_hand_str, contract, is_decl_vuln, 0, sampler, verbose)
        if verbose:
            print("ACE", dummy_hand_str, lefty_hand_str, righty_hand_str, contract)
    elif models.pimc_use_defending and (cardplayer_i == 0):
        from pimc.PIMCDef import BGADefDLL
        pimc[0] = BGADefDLL(models, dummy_hand_str, lefty_hand_str, contract, is_decl_vuln, 0, sampler, verbose)
        if verbose:
            print("PIMC", dummy_hand_str, lefty_hand_str, righty_hand_str, contract)
    else:
        pimc[0] = None

    if ace_mcts_use_defending and (cardplayer_i == 2) and ACEMCTSDefDLL is not None:
        pimc[2] = ACEMCTSDefDLL(models, dummy_hand_str, righty_hand_str, contract, is_decl_vuln, 2, sampler, verbose)
        if verbose:
            print("ACE-MCTS", dummy_hand_str, lefty_hand_str, righty_hand_str, contract)
    elif ace_use_defending and (cardplayer_i == 2) and ACEDefDLL is not None:
        pimc[2] = ACEDefDLL(models, dummy_hand_str, righty_hand_str, contract, is_decl_vuln, 2, sampler, verbose)
        if verbose:
            print("ACE", dummy_hand_str, lefty_hand_str, righty_hand_str, contract)
    elif models.pimc_use_defending and (cardplayer_i == 2):
        from pimc.PIMCDef import BGADefDLL
        pimc[2] = BGADefDLL(models, dummy_hand_str, righty_hand_str, contract, is_decl_vuln, 2, sampler, verbose)
        if verbose:
            print("PIMC", dummy_hand_str, lefty_hand_str, righty_hand_str, contract)
    else:
        pimc[2] = None

    card_players = [
        CardPlayer(models, 0, lefty_hand_str, dummy_hand_str, contract, is_decl_vuln, sampler, pimc[0], dds, verbose),
        CardPlayer(models, 1, dummy_hand_str, decl_hand_str, contract, is_decl_vuln, sampler, pimc[1], dds, verbose),
        CardPlayer(models, 2, righty_hand_str, dummy_hand_str, contract, is_decl_vuln, sampler, pimc[2], dds, verbose),
        CardPlayer(models, 3, decl_hand_str, dummy_hand_str, contract, is_decl_vuln, sampler, pimc[3], dds, verbose)
    ]

    # Clear sample cache at start of new hand
    sampler.clear_sample_cache()

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
                    claimedbydeclarer = (player_i == 3) or (player_i == 1)
                    if (claim <= canclaim):
                        # player_i is relative to declarer
                        if claimedbydeclarer:
                            msg = f"Contract: {contract} Accepted declarers claim of {claim} tricks"
                        else:
                            msg = f"Contract: {contract} Accepted opponents claim of {claim} tricks"
                    else:
                        if claimedbydeclarer:
                            msg = f"Declarer claimed {claim} tricks - rejected {canclaim}"
                        else:
                            msg = f"Opponents claimed {claim} tricks - rejected {canclaim}"
                    print(msg)
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
                # No obvious play, so we roll out
                rollout_states, bidding_scores, c_hcp, c_shp, quality, probability_of_occurence, lead_scores, play_scores, logical_play_scores, discard_scores, worlds = sampler.init_rollout_states(trick_i, player_i, card_players, played_cards, player_cards_played, shown_out_suits, discards, features["aceking"], current_trick, opening_lead52, auction, card_players[player_i].hand_str, card_players[player_i].public_hand_str, [vuln_ns, vuln_ew], models, card_players[player_i].get_random_generator())
                assert rollout_states[0].shape[0] > 0, "No samples for DDSolver"
                
                card_players[player_i].check_pimc_constraints(trick_i, rollout_states, quality)

                card_resp =  card_players[player_i].play_card(trick_i, leader_i, current_trick52, tricks52, rollout_states, worlds, bidding_scores, quality, probability_of_occurence, shown_out_suits, play_status, lead_scores, play_scores, logical_play_scores, discard_scores, features)

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

        if models.pimc_use_declaring or models.pimc_use_defending or ace_use_declaring or ace_use_defending:
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
            if models.pimc_use_defending or ace_use_defending:
                if isinstance(card_players[0], CardPlayer) and card_players[0].pimc:
                    card_players[0].pimc.update_trick_needed()
                if isinstance(card_players[2], CardPlayer) and card_players[2].pimc:
                    card_players[2].pimc.update_trick_needed()
        else:
            card_players[1].n_tricks_taken += 1
            card_players[3].n_tricks_taken += 1
            if models.pimc_use_declaring or ace_use_declaring:
                if isinstance(card_players[3], CardPlayer) and card_players[3].pimc:
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
    # Convert various bid notations to standard format:
    # '--' or 'P' -> 'PASS', 'DB' or 'X' -> 'X', 'RD' or 'XX' -> 'XX'
    def normalize_bid(bid):
        b = bid.upper()
        if b == 'P' or b == '--':
            return 'PASS'
        if b == 'DB':
            return 'X'
        if b == 'RD':
            return 'XX'
        return b
    auction = [normalize_bid(bid) for bid in bids]
    auction = ['PAD_START'] * dealer_i + auction
    return auction

def parse_ctx_to_bids(ctx):
    """
    Parse the ctx (auction context) parameter into a list of bids.

    Accepts multiple formats:
    - New dash-separated: "P-1S-P-3N-P-4S-P-P-P" (single dashes between bids)
    - Old 2-char format: "P 1SP 3NP 4SP P P " or "--1N--3N------" (where -- = Pass)

    Returns a list of bid strings.
    """
    if not ctx:
        return []

    # Detect format: if contains '--' (double dash = Pass in old format), use 2-char chunking
    # New format uses single dashes to separate bids like "P-1N-P-3N"
    # Old format uses '--' for Pass: "--1N--3N--" means "Pass, 1N, Pass, 3N, Pass"

    if '--' in ctx:
        # Old format with '--' for Pass - use 2-char chunking
        ctx_clean = ctx.replace(' ', '')
        bids = [ctx_clean[i:i+2] for i in range(0, len(ctx_clean), 2)]
    elif '-' in ctx:
        # New dash-separated format: "P-1N-P-3N-P-P-P"
        bids = [b.strip() for b in ctx.split('-') if b.strip()]
    elif ' ' in ctx and len(ctx.split()) > 1:
        # Could be space-separated individual bids or the old format
        parts = ctx.split()
        # If parts look like individual bids (1-2 chars each), treat as space-separated
        if all(len(p) <= 2 for p in parts):
            bids = parts
        else:
            # Fall back to 2-char chunking
            ctx_clean = ctx.replace(' ', '')
            bids = [ctx_clean[i:i+2] for i in range(0, len(ctx_clean), 2)]
    else:
        # Concatenated format - split into 2-character chunks
        bids = [ctx[i:i+2] for i in range(0, len(ctx), 2)]

    return bids

def parse_vuln(v):
    """
    Parse vulnerability from absolute format to [vuln_ns, vuln_ew].

    Accepted formats:
    - None, - (empty): No one vulnerable
    - NS, N-S: North-South vulnerable
    - EW, E-W: East-West vulnerable
    - All, Both: Both sides vulnerable

    Returns: [vuln_ns, vuln_ew] where each is True/False
    """
    if v is None:
        v = ''
    v_upper = v.upper().strip()
    vuln_ns = v_upper in ['NS', 'N-S', 'BOTH', 'ALL']
    vuln_ew = v_upper in ['EW', 'E-W', 'BOTH', 'ALL']
    return [vuln_ns, vuln_ew]

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
parser.add_argument("--opponent", default="", help="Filename for configuration pf opponents")
parser.add_argument("--verbose", type=str_to_bool, default=False, help="Output samples and other information during play")
parser.add_argument("--port", type=int, default=8085, help="Port for appserver")
parser.add_argument("--record", type=str_to_bool, default=True, help="Recording of responses")
parser.add_argument("--seed", type=int, default=42, help="Seed for random")
parser.add_argument("--matchpoint", type=str_to_bool, default=None, help="Playing match point")
parser.add_argument("--nolimit", type=str_to_bool, default=False, help="Removed limit on number of requests to the API")

args = parser.parse_args()

configfile = args.config
opponentfile = args.opponent
verbose = args.verbose
port = args.port
record = args.record
matchpoint = args.matchpoint
seed = args.seed
nolimit = args.nolimit

np.set_printoptions(precision=2, suppress=True, linewidth=200)

print(f"{Fore.CYAN}{datetime.datetime.now():%Y-%m-%d %H:%M:%S} gameapi.py - Version {version}{Fore.RESET}")
if util.is_pyinstaller_executable():
    print(f"Running inside a PyInstaller-built executable. {platform.python_version()}")
else:
    print(f"Running in a standard Python environment: {platform.python_version()}")

print(f"Python version: {sys.version}{Fore.RESET} {verbose}")

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
sys.stderr.write(f"NumPy Version : {np.__version__}\n")

configuration = conf.load(configfile)

try:
    if (configuration["models"]['tf_version'] == "2"):
        from nn.models_tf2 import Models
    else: 
        # Default to version 1. of Tensorflow
        from nn.models_tf2 import Models
except KeyError:
        # Default to version 1. of Tensorflow
        from nn.models_tf2 import Models

models = Models.from_conf(configuration, config_path.replace(os.path.sep + "src",""), verbose)
    
print("Config:", configfile)
if opponentfile != "":
    # Override with information from opponent file
    print("Opponent:", opponentfile)
    configuration.read(opponentfile)
    opponents = Opponents.from_conf(configuration, config_path.replace(os.path.sep + "src",""))
    sys.stderr.write(f"Expecting opponent: {opponents.name}\n")

models = Models.from_conf(configuration, config_path.replace(os.path.sep + "src",""))

# Enable model timing for performance analysis
ModelTimer.enabled = True

if verbose:
    print("Loading sampler")
sampler = Sample.from_conf(configuration, verbose)

if sys.platform != 'win32':
    print("Disabling PIMC/BBA as platform is not win32")
    models.pimc_use_declaring = False
    models.pimc_use_defending = False
    #models.use_bba = False
    #models.consult_bba = False
    #models.use_bba_rollout = False
    #models.use_bba_to_count_aces = False

if models.use_bba:
    print("Using BBA for bidding")
else:
    print("Model:   ", os.path.basename(models.bidder_model.model_path))
    print("Opponent:", os.path.basename(models.opponent_model.model_path))

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

if getattr(models, 'ace_use_declaring', False) or getattr(models, 'ace_use_defending', False):
    from ace.ACE import ACEDLL
    ace = ACEDLL(None, None, None, None, None, None, None)
    from ace.ACEDef import ACEDefDLL
    acedef = ACEDefDLL(None, None, None, None, None, None, None, None)
    print(f"ACE enabled. Version {ace.version()}")
    print(f"ACEDef enabled. Version {acedef.version()}")

from ddsolver.ddssolver import DDSSolver
dds_max_threads = configuration.getint('dds', 'dds_max_threads', fallback=0)
dds = DDSSolver(max_threads=dds_max_threads)
print(f"DDSSolver enabled. Version {dds.version()} Max threads {dds_max_threads}")

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

# --- Configure Flask-Limiter ---
def is_internal_request():
    """Exempt internal requests from rate limiting (localhost and private networks)"""
    from flask import request
    remote_addr = request.remote_addr or ''
    # Exempt localhost (IPv4 and IPv6)
    if remote_addr in ('127.0.0.1', '::1', 'localhost'):
        return True
    # Exempt private network ranges (RFC 1918)
    if remote_addr.startswith('192.168.') or \
       remote_addr.startswith('10.') or \
       remote_addr.startswith('172.16.') or \
       remote_addr.startswith('172.17.'):  # Docker default bridge
        return True
    return False

limiter = Limiter(
    app=app,
    key_func=get_remote_address,  # Limits based on the remote IP address
    default_limits=["20000 per day", "5000 per hour", "100 per minute"]
    # storage_uri="memory://" # Default, suitable for single-process test server.
                               # For production with multiple workers, use Redis or Memcached:
                               # "redis://localhost:6379"
)
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
logger.propagate = False  # Prevent log messages from going to the root logger

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
    # --- Get the Referer header ---
    # Use .get() with a default value in case the header is not present
    referrer = request.headers.get("Referer", "None") # Use "None" or "" or None as default

    # Log method, URL, and Referer
    # You can format this however you like
    logger.info(f"Request: {request.method} {request.url} Referer: {referrer}")

    # Log POST body (unchanged from your original)
    if request.method == "POST":
        # Consider potential security implications and size of logging raw body data
        try:
            # Decoding might fail if data isn't valid UTF-8, handle potential errors
            # Also consider limiting the size logged
            body_data = request.get_data(as_text=True)
            max_log_len = 500 # Limit logged body size
            logged_body = body_data[:max_log_len] + ('...' if len(body_data) > max_log_len else '')
            logger.info(f"Body: {logged_body}")
        except Exception as e:
            logger.warning(f"Could not decode or log request body: {e}")
            # Fallback to logging raw bytes info if needed
            raw_body = request.get_data()
            logger.info(f"Body (raw bytes, size): {len(raw_body)}")

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
        # First we extract our hand
        hand = request.args.get("hand").replace('_','.').upper()
        if 'X' in hand:
            if '8' in hand or '9' in hand:
                hand = replace_x(hand,get_random_generator(hand), "...", [], models.n_cards_play)
            else:
                hand = replace_x(hand,get_random_generator(hand), "...", [], models.n_cards_bidding)
        seat = request.args.get("seat")
        # Vulnerability (absolute format: None, NS, EW, All/Both)
        v = request.args.get("vul")
        vuln = parse_vuln(v)
        # And finally the bidding, where we deduct dealer and our position
        dealer = request.args.get("dealer")
        dealer_i = dealer_enum[dealer]
        position_i = dealer_enum[seat]
        ctx = request.args.get("ctx")
        bids = parse_ctx_to_bids(ctx)
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
            explanations = None
        else:

            hint_bot = BotBid(vuln, hand, models, sampler, position_i, dealer_i, dds, False, verbose)
            explanations, bba_controlled, preempted = hint_bot.explain_auction(auction)
            hint_bot.bba_is_controlling = bba_controlled
        with model_lock_bid:
            bid = hint_bot.bid(auction)

        result = bid.to_dict()
        if not details:
            if "candidates" in result: del result["candidates"]
            if "samples" in result: del result["samples"]
            if "shape" in result: del result["shape"]
            if "hcp" in result: del result["hcp"]
        else:
            result["explanations"] = explanations

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
        # Vulnerability (absolute format: None, NS, EW, All/Both)
        v = request.args.get("vul")
        vuln = parse_vuln(v)
        # And finally the dealer and bidding
        dealer = request.args.get("dealer")
        dealer_i = dealer_enum[dealer]
        position = dealer_enum[seat]
        ctx = request.args.get("ctx")
        bids = parse_ctx_to_bids(ctx)
        auction = create_auction(bids, dealer_i)
        contract = bidding.get_contract(auction)
        if contract is None:
            result = {"error": "No contract found - auction context (ctx) is required"}
            return json.dumps(result), 400
        decl_i = bidding.get_decl_i(contract)
        if "NESW"[(decl_i + 1) % 4] != seat:
            result = {"message":"Not this player to lead"}
            print(result)
            return json.dumps(result)

        # Allow verbose to be enabled per-request via query parameter
        request_verbose = request.args.get("verbose", "").lower() in ("true", "1", "yes")
        effective_verbose = verbose or request_verbose

        # Find ace and kings
        aceking = {}
        if models.use_bba_to_count_aces:
            from bba.BBA import BBABotBid
            bba_bot = BBABotBid(models.bba_our_cc, models.bba_their_cc, position, hand, vuln, dealer_i, models.matchpoint, effective_verbose)
            aceking = bba_bot.find_aces(auction)
            if verbose:
                bba_bot.get_sample(auction)

        hint_bot = BotLead(vuln, hand, models, sampler, position, dealer_i, dds, effective_verbose)
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
@limiter.limit("10000/hour;100/minute", exempt_when=is_internal_request)
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

        if len(dummy_str) != 16:
            result = {"message":"Dummy should have 13 cards"}
            print(result)
            return json.dumps(result)

        if len(hand_str) != 16:
            result = {"message":"Hand should have 13 cards"}
            print(result)
            return json.dumps(result)

        seat = request.args.get("seat")
        # Vulnerability (absolute format: None, NS, EW, All/Both)
        v = request.args.get("vul")
        vuln = parse_vuln(v)
        # And finally the bidding, where we deduct dealer and our position
        dealer = request.args.get("dealer")
        dealer_i = dealer_enum[dealer]
        position_i = dealer_enum[seat]
        ctx = request.args.get("ctx")
        bids = parse_ctx_to_bids(ctx)
        # Validate number of cards played according to position
        auction = create_auction(bids, dealer_i)
        contract = bidding.get_contract(auction)
        if contract is None:
            result = {"error": "No contract found - auction context (ctx) is required"}
            return json.dumps(result), 400
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

        # Allow verbose to be enabled per-request via query parameter
        request_verbose = request.args.get("verbose", "").lower() in ("true", "1", "yes")
        effective_verbose = verbose or request_verbose

        # Find ace and kings, when defending
        # Find ace and kings
        features = {}
        aceking = {}
        if models.use_bba_to_count_aces:
            from bba.BBA import BBABotBid
            bba_bot = BBABotBid(models.bba_our_cc, models.bba_their_cc, position_i, "KJ53.KJ7.AT92.K5", vuln, dealer_i, models.matchpoint, effective_verbose)
            aceking = bba_bot.find_aces(auction)
            features["aceking"] = aceking
            #bba_bot.get_sample(auction)
            bba_bot = BBABotBid(models.bba_our_cc, models.bba_their_cc, position_i, "KJ53.KJ7.AT92.K5", vuln, dealer_i, models.matchpoint, effective_verbose)
            explanation, _, preempted = bba_bot.explain_auction(auction)
            features["Explanation"] = explanation
            features["preempted"] = preempted
        else:
            features["aceking"] = aceking

        # Play
        with model_lock_play:
            card_resp, player_i, msg =  play_api(dealer_i, vuln[0], vuln[1], hands, models, sampler, contract, strain_i, decl_i, auction, cards, cardplayer, False, features, effective_verbose)
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
    auction = [bid.upper().replace('--', "PASS").replace('DB', 'X').replace('RD', 'XX').replace("NT","N") for bid in auction]

    vuln_ns = vuln_input == 'NS' or vuln_input == 'ALL'
    vuln_ew = vuln_input == 'EW' or vuln_input == 'ALL'

    vuln = [vuln_ns, vuln_ew]

    if models.use_bba:
        from bba.BBA import BBABotBid
        hint_bot = BBABotBid(models.bba_our_cc, models.bba_their_cc, position_i, hand, vuln, dealer_i, models.matchpoint, verbose)
    else:
        hint_bot = BotBid(vuln, hand, models, sampler, position_i, dealer_i, dds, False, verbose)
        explanations, bba_controlled, preempted = hint_bot.explain_auction(auction, dealer_i)
        hint_bot.bba_is_controlling = bba_controlled
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
    # Vulnerability (absolute format: None, NS, EW, All/Both)
    v = request.args.get("vul")
    vuln = parse_vuln(v)
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
    bids = parse_ctx_to_bids(ctx)

    auction = create_auction(bids, dealer_i)

    explanation, alert = bot.explain_last_bid(auction)
    
    result = {"explanation": explanation, "Alert": alert} # explaination
    print(f'Request took {(time.time() - t_start):0.2f} seconds')       

    return json.dumps(result)
@app.route('/explain_auction')
def explain_auction():
    t_start = time.time()
    from bba.BBA import BBABotBid
    # First we extract the hands and seat
    seat = request.args.get("seat")
    # Vulnerability (absolute format: None, NS, EW, All/Both)
    v = request.args.get("vul")
    vuln = parse_vuln(v)
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
    bids = parse_ctx_to_bids(ctx)

    auction = create_auction(bids, dealer_i)

    explanation, bba_controlled, preempted = bot.explain_auction(auction)
    
    # Create the HTML list
    html_list = '<ul>\n'  # Start the unordered list
    for item in explanation:
        key, value = item  # Unpack the tuple
        html_list += f'  <li>{key} {value}</li>\n'  # Add formatted tuple as list item
    html_list += '</ul>'  # Close the unordered list


    result = {"explanation": html_list, "bba_controlled": bba_controlled} # explaination
    print(f'Request took {(time.time() - t_start):0.2f} seconds')       

    return json.dumps(result)
@app.route('/bids')
@limiter.limit("10000/hour;100/minute", exempt_when=is_internal_request)
def bids():
    t_start = time.time()
    base_path = os.getenv('BEN_HOME') or '..'
    file_us = request.args.get("file_us")
    file_them = request.args.get("file_them")
    if not file_us:
        file_us = models.bba_our_cc
    else:
        file_us = os.path.join(base_path,"BBA/CC/" + file_us)
    if not file_them:
        file_them = models.bba_their_cc
    else:
        file_them = os.path.join(base_path,"BBA/CC/" + file_them)
        
    from bba.BBA import BBABotBid
    if verbose:
        print("file_us", file_us, "file_them", file_them)
    dealer_i = 0
    position_i = 0
    mp = False
    vuln = [False, False]
    bot = BBABotBid(file_us, file_them, position_i, "KJ53.KJ7.AT92.K5", vuln, dealer_i, mp, verbose)
    ctx = request.args.get("ctx").replace('*','').replace("XX","Rd").replace("X","Db").replace('-','').upper().replace("P","--")
    # Split the string into chunks of every second character
    bids = [ctx[i:i+2] for i in range(0, len(ctx), 2)]

    auction = create_auction(bids, dealer_i)
    print("auction", auction, ctx)

    result = bot.list_bids(auction)
    
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
        # Vulnerability (absolute format: None, NS, EW, All/Both)
        v = request.args.get("vul")
        vuln = parse_vuln(v)
        # And finally we deduct our position
        position_i = dealer_enum[seat]
        X = get_binary_contract(position_i, vuln, hand_str, dummy_str, models.n_cards_bidding)
        result = {}
        with model_lock_bid:
            if verbose:
                print(position_i, vuln, hand_str, dummy_str)
                print(X)
            contract_id = models.contract_model.pred_fun(X)
            if hasattr(contract_id, 'numpy'):
                contract_id = contract_id.numpy()
            for i in range(len(contract_id[0])):
                if contract_id[0][i] > 0.05:
                    y = np.zeros(5)
                    suit = bidding.ID2BID[i][1]
                    strain_i = 'NSHDC'.index(suit)
                    y[strain_i] = 1
                    Xt = [np.concatenate((X[0], y), axis=0)]
                    tricks = models.trick_model.pred_fun(Xt)
                    if hasattr(tricks, 'numpy'):
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
        claim = int(request.args.get("tricks"))
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
        # Vulnerability (absolute format: None, NS, EW, All/Both)
        v = request.args.get("vul")
        vuln = parse_vuln(v)
        # And finally the bidding, where we deduct dealer and our position
        dealer = request.args.get("dealer")
        dealer_i = dealer_enum[dealer]
        position_i = dealer_enum[seat]
        ctx = request.args.get("ctx")
        bids = parse_ctx_to_bids(ctx)
        # Validate number of cards played according to position
        auction = create_auction(bids, dealer_i)
        contract = bidding.get_contract(auction)
        if contract is None:
            result = {"error": "No contract found - auction context (ctx) is required"}
            return json.dumps(result), 400
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

# Optional: Custom error handler for rate limit exceeded (HTTP 429)
@app.errorhandler(429)
def ratelimit_handler(e):
    # The `e.description` will contain the limit that was hit.
    return jsonify(error="Rate limit exceeded. Please try again later.", limit=e.description), 429

@app.route('/autoplay')
def autoplay():
    """
    Auto-play a complete board with all 4 hands played by BEN.

    Parameters:
        deal (str): PBN deal string (required) - e.g. "862.62.AQT52.A96 AQJT9.Q875.97.K7 7543.AT943.8.JT8 K.KJ.KJ643.Q5432"
        board (int): Board number 1-16 for dealer/vulnerability (default: 1)
        dealer (str): Optional dealer override (N/E/S/W)
        vul (str): Optional vulnerability override (None/NS/EW/Both)
        bidding_only (str): If "True", return after bidding without card play (default: "False")

    Returns:
        JSON with auction, play, contract, tricks, and score
        If bidding_only=True: JSON with auction, contract, declarer, opening_leader (no play)
    """
    try:
        t_start = time.time()
        ModelTimer.reset()  # Reset timing stats for this request

        # Get parameters
        board_number = request.args.get("board", 1, type=int)
        deal_str = request.args.get("deal")
        dealer_param = request.args.get("dealer")
        vuln_param = request.args.get("vul")
        bidding_only = request.args.get("bidding_only", "False")
        fixed_auction_param = request.args.get("auction")  # Optional: comma-separated bids to use instead of generating

        if deal_str is None:
            return jsonify({"error": "'deal' parameter required"}), 400

        # URL encoding may have converted spaces to + signs
        deal_str = deal_str.replace('+', ' ')

        # Set random seed based on deal hash (same as game.py)
        # This ensures reproducible card play that matches game.py
        hash_integer = calculate_seed(deal_str)
        print(f"[Autoplay] Setting seed based on deal hash: {hash_integer}")
        np.random.seed(hash_integer)
        tf.random.set_seed(hash_integer)

        mp = models.matchpoint      
        if request.args.get("tournament"):
            mp = request.args.get("tournament").lower() == "mp"
            models.matchpoint = mp

        # Get dealer and vulnerability from parameters or board number
        if dealer_param:
            dealer = dealer_param.upper()
        else:
            auction_str = deck52.board_dealer_vuln(board_number)
            parts = auction_str.split()
            dealer = parts[0]

        if vuln_param:
            vuln_str = vuln_param
            vuln_ns = vuln_str.upper() in ['NS', 'N-S', 'BOTH', 'ALL']
            vuln_ew = vuln_str.upper() in ['EW', 'E-W', 'BOTH', 'ALL']
        else:
            auction_str = deck52.board_dealer_vuln(board_number)
            parts = auction_str.split()
            vuln_str = parts[1] if len(parts) > 1 else 'None'
            vuln_ns = vuln_str in ['N-S', 'Both']
            vuln_ew = vuln_str in ['E-W', 'Both']

        dealer_i = dealer_enum[dealer]
        hands = deal_str.split()
        vuln = [vuln_ns, vuln_ew]  # Set vuln early so it's available for both bidding and fixed auction paths

        if verbose:
            print(f"[Autoplay] Board: {board_number}, Deal hash: {hash_integer}")
            print(f"[Autoplay] Deal: {deal_str}")
            print(f"[Autoplay] Dealer: {dealer}, Vuln: {vuln_str}")

        # Initialize bid explanations list
        bid_explanations = []

        # Check if a fixed auction was provided (for declare_only/defend_only modes)
        if fixed_auction_param:
            # Parse the fixed auction - expect comma-separated bids or JSON array
            # Format: "Pass,1C,1S,Pass,2C,Pass,Pass,Pass" or similar
            if fixed_auction_param.startswith('['):
                import json
                fixed_bids = json.loads(fixed_auction_param)
            else:
                fixed_bids = [b.strip() for b in fixed_auction_param.split(',')]

            # Normalize bid names
            normalized_bids = []
            for bid in fixed_bids:
                bid_upper = bid.upper().strip()
                if bid_upper in ['PASS', 'P', '--']:
                    normalized_bids.append('PASS')
                elif bid_upper in ['X', 'DBL', 'DOUBLE']:
                    normalized_bids.append('X')
                elif bid_upper in ['XX', 'RDBL', 'REDOUBLE']:
                    normalized_bids.append('XX')
                else:
                    # Convert NT variations
                    bid_normalized = bid_upper.replace('NT', 'N')
                    normalized_bids.append(bid_normalized)

            # Build full auction with PAD_START tokens
            auction = ['PAD_START'] * dealer_i + normalized_bids
            print(f"[Autoplay] Using fixed auction: {normalized_bids}")

            # For fixed auctions, we don't have explanations from the original generation
            # Create empty explanation entries
            current_bidder = dealer_i
            for bid in normalized_bids:
                bid_explanations.append({
                    "bid": bid,
                    "player": ['N','E','S','W'][current_bidder],
                    "explanation": "",
                    "alert": False
                })
                current_bidder = (current_bidder + 1) % 4

        else:
            # Run bidding for all 4 positions
            # Start with PAD_START tokens for positions before dealer
            auction = ['PAD_START'] * dealer_i
            current_bidder = dealer_i
            pass_count = 0

            while True:
                hand = hands[current_bidder]
                vuln = [vuln_ns, vuln_ew]

                if models.use_bba:
                    from bba.BBA import BBABotBid
                    hint_bot = BBABotBid(models.bba_our_cc, models.bba_their_cc, current_bidder, hand, vuln, dealer_i, models.matchpoint, False)
                else:
                    hint_bot = BotBid(vuln, hand, models, sampler, current_bidder, dealer_i, dds, False, False)
                    # Initialize bbabot by calling explain_auction (needed for explain_last_bid)
                    try:
                        hint_bot.explain_auction(auction)
                    except Exception as e:
                        print(f"[Autoplay] Failed to initialize bbabot: {e}")

                with model_lock_bid:
                    bid_resp = hint_bot.bid(auction)

                bid = bid_resp.bid
                auction.append(bid)

                # Get explanation for this bid
                explanation = ""
                alert = False
                try:
                    if models.use_bba:
                        explanation, alert = hint_bot.explain_last_bid(auction)
                    else:
                        # Re-initialize bbabot with updated auction and get explanation
                        hint_bot.explain_auction(auction)
                        explanation, alert = hint_bot.bbabot.explain_last_bid(auction)
                    if verbose:
                        print(f"[Autoplay] {['N','E','S','W'][current_bidder]} bids {bid}: explanation='{explanation}', alert={alert}")
                except Exception as e:
                    print(f"[Autoplay] Could not get explanation for {['N','E','S','W'][current_bidder]}'s {bid}: {e}")

                bid_explanations.append({
                    "bid": bid,
                    "player": ['N','E','S','W'][current_bidder],
                    "explanation": explanation or "",
                    "alert": alert
                })

                # Check for auction end
                if bid == 'PASS':
                    pass_count += 1
                else:
                    pass_count = 0

                if pass_count >= 3 and len(auction) >= 4:
                    break
                if len(auction) == 4 and all(b == 'PASS' for b in auction):
                    break

                current_bidder = (current_bidder + 1) % 4

        # Get contract
        contract = bidding.get_contract(auction)

        # Handle passed out
        if contract is None:
            # Filter out PAD_START from auction for output
            auction_output = [bid for bid in auction if bid != 'PAD_START']
            result = {
                "deal": deal_str,
                "dealer": dealer,
                "vulnerability": vuln_str,
                "auction": auction_output,
                "auction_with_explanations": bid_explanations,
                "contract": "PASS",
                "declarer": None,
                "tricks": 0,
                "score": 0,
                "ns_score": 0,
                "play": [],
                "elapsed": round(time.time() - t_start, 2)
            }
            print(f'[Autoplay] Passed out - took {(time.time() - t_start):0.2f} seconds')
            print(ModelTimer.get_summary())
            return jsonify(result)

        strain_i = bidding.get_strain_i(contract)
        decl_i = bidding.get_decl_i(contract)
        level = int(contract[0])
        is_decl_vuln = [vuln_ns, vuln_ew, vuln_ns, vuln_ew][decl_i]

        if verbose:
            print(f"[Autoplay] Contract: {contract} by {['N','E','S','W'][decl_i]}")

        # Print PBN after bidding (for use with game.py verification)
        auction_for_pbn = [bid for bid in auction if bid != 'PAD_START']
        print("\n=== PBN AFTER BIDDING (use as input to game.py) ===")
        print(f'[Event "BEN Autoplay"]')
        print(f'[Board "{board_number}"]')
        print(f'[Dealer "{dealer}"]')
        print(f'[Vulnerable "{vuln_str}"]')
        print(f'[Deal "N:{deal_str}"]')
        print(f'[Declarer "{["N","E","S","W"][decl_i]}"]')
        print(f'[Contract "{contract}"]')
        print(f'[Auction "{dealer}"]')
        print(' '.join(auction_for_pbn))
        print("=== END PBN ===\n")

        # If bidding_only mode, return now without card play
        # Check for various true values (True, true, TRUE, 1, yes)
        bidding_only_value = str(bidding_only).upper().strip()
        is_bidding_only = bidding_only_value in ["TRUE", "1", "YES"]

        if verbose:
            print(f"[Autoplay] bidding_only parameter: '{bidding_only}' -> is_bidding_only: {is_bidding_only}")

        if is_bidding_only:
            # Calculate opening leader (LHO of declarer)
            leader_i = (decl_i + 1) % 4
            opening_leader = ['N','E','S','W'][leader_i]

            result = {
                "deal": deal_str,
                "dealer": dealer,
                "vulnerability": vuln_str,
                "auction": auction_for_pbn,
                "auction_with_explanations": bid_explanations,
                "contract": contract,
                "declarer": ['N','E','S','W'][decl_i],
                "opening_leader": opening_leader,
                "bidding_only": True,
                "elapsed": round(time.time() - t_start, 2)
            }
            print(f'[Autoplay] Bidding only: {contract} by {["N","E","S","W"][decl_i]} - took {(time.time() - t_start):0.2f} seconds')
            print(ModelTimer.get_summary())
            return jsonify(result)

        # Run card play
        play_cards = []

        # Opening lead from LHO of declarer
        # leader_nesw is in NESW coordinates (for BotLead), leader_i is in CardPlayer coordinates (for card play tracking)
        leader_nesw = (decl_i + 1) % 4
        leader_i = 0  # Player 0 is always lefty (opening leader) in CardPlayer coordinates

        # Get opening lead
        # Need to find aces first if using BBA
        aceking = {}
        if models.use_bba:
            from bba.BBA import BBABotBid
            bba_bot = BBABotBid(models.bba_our_cc, models.bba_their_cc, leader_nesw, hands[leader_nesw], vuln, dealer_i, models.matchpoint, False)
            aceking = bba_bot.find_aces(auction)

        with model_lock_play:
            lead_bot = BotLead(vuln, hands[leader_nesw], models, sampler, leader_nesw, dealer_i, dds, False)
            lead_resp = lead_bot.find_opening_lead(auction, aceking)

        opening_lead = lead_resp.card.code()
        play_cards.append(deck52.decode_card(opening_lead))

        # Print play header for comparison with game.py
        print(f"\n=== PLAY (compare with game.py) ===")
        print(f"Opening leader: {['N','E','S','W'][leader_nesw]} (LHO of declarer {['N','E','S','W'][decl_i]})")
        print(f"Trick:   S   W   N   E   Winner")

        if verbose:
            print(f"[Autoplay] Opening lead: {deck52.decode_card(opening_lead)}")

        # Setup card players
        lefty_hand_str = hands[(decl_i + 1) % 4]
        dummy_hand_str = hands[(decl_i + 2) % 4]
        righty_hand_str = hands[(decl_i + 3) % 4]
        decl_hand_str = hands[decl_i]

        # Create PIMC/ACE objects based on model settings (following game.py pattern)
        pimc = [None, None, None, None]

        # ACE takes priority over PIMC if both are enabled
        if getattr(models, 'ace_use_declaring', False):
            from ace.ACE import ACEDLL
            declarer_pimc = ACEDLL(models, dummy_hand_str, decl_hand_str, contract, is_decl_vuln, sampler, verbose)
            pimc[1] = declarer_pimc
            pimc[3] = declarer_pimc
            if verbose:
                print(f"[Autoplay] ACE (declarer) enabled")
        elif getattr(models, 'pimc_use_declaring', False):
            from pimc.PIMC import BGADLL
            declarer_pimc = BGADLL(models, dummy_hand_str, decl_hand_str, contract, is_decl_vuln, sampler, verbose)
            pimc[1] = declarer_pimc
            pimc[3] = declarer_pimc
            if verbose:
                print(f"[Autoplay] PIMC (declarer) enabled")

        if getattr(models, 'ace_use_defending', False):
            from ace.ACEDef import ACEDefDLL
            pimc[0] = ACEDefDLL(models, dummy_hand_str, lefty_hand_str, contract, is_decl_vuln, 0, sampler, verbose)
            pimc[2] = ACEDefDLL(models, dummy_hand_str, righty_hand_str, contract, is_decl_vuln, 2, sampler, verbose)
            if verbose:
                print(f"[Autoplay] ACE (defender) enabled")
        elif getattr(models, 'pimc_use_defending', False):
            from pimc.PIMCDef import BGADefDLL
            pimc[0] = BGADefDLL(models, dummy_hand_str, lefty_hand_str, contract, is_decl_vuln, 0, sampler, verbose)
            pimc[2] = BGADefDLL(models, dummy_hand_str, righty_hand_str, contract, is_decl_vuln, 2, sampler, verbose)
            if verbose:
                print(f"[Autoplay] PIMC (defender) enabled")

        card_players = [
            CardPlayer(models, 0, lefty_hand_str, dummy_hand_str, contract, is_decl_vuln, sampler, pimc[0], dds, verbose),
            CardPlayer(models, 1, dummy_hand_str, decl_hand_str, contract, is_decl_vuln, sampler, pimc[1], dds, verbose),
            CardPlayer(models, 2, righty_hand_str, dummy_hand_str, contract, is_decl_vuln, sampler, pimc[2], dds, verbose),
            CardPlayer(models, 3, decl_hand_str, dummy_hand_str, contract, is_decl_vuln, sampler, pimc[3], dds, verbose)
        ]

        # Clear sample cache at start of new hand
        sampler.clear_sample_cache()

        # Play all 13 tricks
        player_cards_played = [[] for _ in range(4)]
        player_cards_played52 = [[] for _ in range(4)]
        shown_out_suits = [set() for _ in range(4)]
        discards = [set() for _ in range(4)]
        tricks52 = []
        current_trick52 = [opening_lead]
        current_trick = [deck52.card52to32(opening_lead)]

        # Features dict for play_card
        features = {"aceking": aceking}

        # Opening lead is already in current_trick/current_trick52
        # player_cards_played is only updated AFTER a trick completes (following game.py pattern)
        opening_lead32 = deck52.card52to32(opening_lead)

        # Update card players with opening lead
        card_players[0].hand52[opening_lead] -= 1
        for i, card_player in enumerate(card_players):
            card_player.set_real_card_played(opening_lead, 0, openinglead=True)
            card_player.set_card_played(trick_i=0, leader_i=0, i=0, card=opening_lead32)

        current_player_i = 1  # Dummy plays next after opening lead
        trick_i = 0

        for _ in range(51):  # 51 more cards to play
            cardplayer_i = current_player_i

            # Get play status
            play_status = get_play_status(card_players[cardplayer_i].hand52, current_trick52, strain_i)

            # Check for forced play (singleton in suit)
            card_resp = None
            if play_status == "Forced" and models.autoplaysingleton:
                card = get_singleton(card_players[cardplayer_i].hand52, current_trick52)
                card_resp = CardResp(
                    card=Card.from_code(card),
                    candidates=[],
                    samples=[],
                    shape=-1,
                    hcp=-1,
                    quality=None,
                    who="Forced",
                    claim=-1
                )

            # Check for equivalent cards (follow with same value cards)
            if card_resp is None and play_status == "Follow":
                high, low = get_possible_cards(card_players[cardplayer_i].hand52, current_trick52)
                if high != -1:
                    card = high if cardplayer_i == 3 else low
                    card_resp = CardResp(
                        card=Card.from_code(card),
                        candidates=[],
                        samples=[],
                        shape=-1,
                        hcp=-1,
                        quality=None,
                        who="Follow",
                        claim=-1
                    )

            # Need to do rollout sampling
            if card_resp is None:
                played_cards = [card for row in player_cards_played52 for card in row] + current_trick52

                try:
                    with model_lock_play:
                        rollout_states, bidding_scores, c_hcp, c_shp, quality, probability_of_occurence, lead_scores, play_scores, logical_play_scores, discard_scores, worlds = sampler.init_rollout_states(
                            trick_i, cardplayer_i, card_players, played_cards, player_cards_played,
                            shown_out_suits, discards, features["aceking"], current_trick, opening_lead,
                            auction, card_players[cardplayer_i].hand_str, card_players[cardplayer_i].public_hand_str,
                            [vuln_ns, vuln_ew], models, card_players[cardplayer_i].get_random_generator()
                        )

                        card_players[cardplayer_i].check_pimc_constraints(trick_i, rollout_states, quality)

                        card_resp = card_players[cardplayer_i].play_card(
                            trick_i, leader_i, current_trick52, tricks52,
                            rollout_states, worlds, bidding_scores, quality, probability_of_occurence,
                            shown_out_suits, play_status, lead_scores, play_scores,
                            logical_play_scores, discard_scores, features
                        )
                except Exception as e:
                    # Log the actual error for debugging
                    print(f"[Autoplay] Error in card play at trick {trick_i}, player {cardplayer_i}: {type(e).__name__}: {e}")

                    # Check if this is a PIMC no-playout error - fallback to first legal card
                    is_pimc_error = (
                        (PIMCNoPlayoutErrorDeclarer and isinstance(e, PIMCNoPlayoutErrorDeclarer)) or
                        (PIMCNoPlayoutErrorDefender and isinstance(e, PIMCNoPlayoutErrorDefender)) or
                        "PIMCNoPlayoutError" in type(e).__name__
                    )
                    if is_pimc_error:
                        print(f"[Autoplay] PIMC fallback for player {cardplayer_i} at trick {trick_i}: {e}")
                        # Pick first legal card from hand
                        hand52 = card_players[cardplayer_i].hand52
                        if current_trick52:  # Must follow suit if possible
                            lead_suit = current_trick52[0] // 13
                            suit_cards = [i for i in range(lead_suit * 13, (lead_suit + 1) * 13) if hand52[i] == 1]
                            if suit_cards:
                                fallback_card = suit_cards[0]
                            else:
                                # No cards in lead suit, play any card
                                fallback_card = next(i for i in range(52) if hand52[i] == 1)
                        else:
                            # Leading - play any card
                            fallback_card = next(i for i in range(52) if hand52[i] == 1)
                        card_resp = CardResp(
                            card=Card.from_code(fallback_card),
                            candidates=[],
                            samples=[],
                            shape=-1,
                            hcp=-1,
                            quality=None,
                            who="PIMC Fallback",
                            claim=-1
                        )
                    else:
                        raise  # Re-raise non-PIMC errors

            card52 = card_resp.card.code()
            card32 = deck52.card52to32(card52)
            play_cards.append(deck52.decode_card(card52))

            # Update card players with played card
            for card_player in card_players:
                card_player.set_real_card_played(card52, cardplayer_i)
                card_player.set_card_played(trick_i=trick_i, leader_i=leader_i, i=cardplayer_i, card=card32)

            card_players[cardplayer_i].set_own_card_played52(card52)
            if cardplayer_i == 1:
                for i in [0, 2, 3]:
                    card_players[i].set_public_card_played52(card52)
            if cardplayer_i == 3:
                card_players[1].set_public_card_played52(card52)

            # Add to current trick (player_cards_played is updated only after trick completes)
            current_trick52.append(card52)
            current_trick.append(card32)

            # Check for shown out
            if len(current_trick52) > 1:
                lead_card32 = current_trick[0]
                lead_suit32 = lead_card32 // 8
                card_suit32 = card32 // 8
                if card_suit32 != lead_suit32:  # Player showed out of lead suit
                    shown_out_suits[cardplayer_i].add(lead_suit32)  # Add the lead suit they can't follow
                    discards[cardplayer_i].add((trick_i, card32))

            if len(current_trick52) == 4:
                # Determine trick winner
                # Note: strain_i from bidding uses N=0,S=1,H=2,D=3,C=4
                # But deck52.get_trick_winner_i expects suit indices S=0,H=1,D=2,C=3
                # The formula (strain_i - 1) % 5 converts correctly (N=0 becomes 4 which never matches any suit)
                winner_i = deck52.get_trick_winner_i(current_trick52, (strain_i - 1) % 5)
                winner_pos = (leader_i + winner_i) % 4

                if winner_pos in [0, 2]:  # Defenders (lefty/righty)
                    card_players[0].n_tricks_taken += 1
                    card_players[2].n_tricks_taken += 1
                    # Update PIMC/ACE trick needed for defenders
                    if getattr(models, 'pimc_use_defending', False) or getattr(models, 'ace_use_defending', False):
                        if card_players[0].pimc:
                            card_players[0].pimc.update_trick_needed()
                        if card_players[2].pimc:
                            card_players[2].pimc.update_trick_needed()
                else:  # Declarers (dummy/declarer)
                    card_players[1].n_tricks_taken += 1
                    card_players[3].n_tricks_taken += 1
                    # Update PIMC/ACE trick needed for declarer
                    if getattr(models, 'pimc_use_declaring', False) or getattr(models, 'ace_use_declaring', False):
                        if card_players[3].pimc:
                            card_players[3].pimc.update_trick_needed()

                tricks52.append(current_trick52)

                # Print trick for comparison with game.py
                # Cards are in CardPlayer order: lefty(0), dummy(1), righty(2), declarer(3)
                # Convert to NESW order for PBN output
                cardplayer_to_nesw = [(decl_i + 1) % 4, (decl_i + 2) % 4, (decl_i + 3) % 4, decl_i]
                trick_cards_nesw = [''] * 4
                for i, c52 in enumerate(current_trick52):
                    nesw_pos = cardplayer_to_nesw[(leader_i + i) % 4]
                    trick_cards_nesw[nesw_pos] = deck52.decode_card(c52)
                # Print in order starting from South (opening leader for 3N by E is S)
                winner_nesw = cardplayer_to_nesw[winner_pos]
                print(f"Trick {trick_i + 1}: {trick_cards_nesw[2]:>3} {trick_cards_nesw[3]:>3} {trick_cards_nesw[0]:>3} {trick_cards_nesw[1]:>3}  Won by {['N','E','S','W'][winner_nesw]}")

                # Update player_cards_played AFTER trick completes (following game.py pattern)
                for i, c32 in enumerate(current_trick):
                    player_cards_played[(leader_i + i) % 4].append(c32)
                for i, c52 in enumerate(current_trick52):
                    player_cards_played52[(leader_i + i) % 4].append(c52)

                # Reset PIMC/ACE trick state if enabled
                if getattr(models, 'pimc_use_declaring', False) or getattr(models, 'pimc_use_defending', False) or getattr(models, 'ace_use_declaring', False) or getattr(models, 'ace_use_defending', False):
                    for card_player in card_players:
                        if card_player.pimc:
                            card_player.pimc.reset_trick()

                # Initialize x_play for next trick (following game.py pattern)
                # Skip for the last trick (trick_i == 12) since there's no trick 13
                if trick_i < 12:
                    # Initialize hands
                    for i, c32 in enumerate(current_trick):
                        card_players[(leader_i + i) % 4].x_play[:, trick_i + 1, 0:32] = card_players[(leader_i + i) % 4].x_play[:, trick_i, 0:32]
                        card_players[(leader_i + i) % 4].x_play[:, trick_i + 1, 0 + c32] -= 1

                    # Initialize public hands
                    for i in (0, 2, 3):
                        card_players[i].x_play[:, trick_i + 1, 32:64] = card_players[1].x_play[:, trick_i + 1, 0:32]
                    card_players[1].x_play[:, trick_i + 1, 32:64] = card_players[3].x_play[:, trick_i + 1, 0:32]

                    for card_player in card_players:
                        # Initialize last trick
                        for i, c32 in enumerate(current_trick):
                            card_player.x_play[:, trick_i + 1, 64 + i * 32 + c32] = 1

                        # Initialize last trick leader
                        card_player.x_play[:, trick_i + 1, 288 + leader_i] = 1

                        # Initialize level
                        card_player.x_play[:, trick_i + 1, 292] = level

                        # Initialize strain
                        card_player.x_play[:, trick_i + 1, 293 + strain_i] = 1

                if verbose:
                    # Convert winner_pos (CardPlayer index: 0=lefty, 1=dummy, 2=righty, 3=declarer) to NESW for display
                    cardplayer_to_nesw = [(decl_i + 1) % 4, (decl_i + 2) % 4, (decl_i + 3) % 4, decl_i]
                    winner_nesw = cardplayer_to_nesw[winner_pos]
                    print(f"[Autoplay] Trick {trick_i+1}: {[deck52.decode_card(c) for c in current_trick52]} - won by {['N','E','S','W'][winner_nesw]}")

                # Setup for next trick
                trick_i += 1
                leader_i = winner_pos
                current_trick52 = []
                current_trick = []
                current_player_i = winner_pos
            else:
                current_player_i = (leader_i + len(current_trick52)) % 4

        # Final tricks count from declarer's perspective
        decl_tricks = card_players[3].n_tricks_taken

        # Calculate score
        from scoring import score as calculate_score
        score = calculate_score(contract, is_decl_vuln, decl_tricks)
        ns_score = score if decl_i in [0, 2] else -score

        # Filter out PAD_START from auction for output
        auction_output = [bid for bid in auction if bid != 'PAD_START']
        result = {
            "deal": deal_str,
            "dealer": dealer,
            "vulnerability": vuln_str,
            "auction": auction_output,
            "auction_with_explanations": bid_explanations,
            "contract": contract,
            "declarer": ['N','E','S','W'][decl_i],
            "tricks": decl_tricks,
            "score": score,
            "ns_score": ns_score,
            "play": play_cards,
            "elapsed": round(time.time() - t_start, 2)
        }

        print(f"=== END PLAY ===")
        print(f'[Autoplay] {contract} by {["N","E","S","W"][decl_i]}, {decl_tricks} tricks, score={score} - took {(time.time() - t_start):0.2f} seconds')

        # Print PBN to console for verification with game.py
        print("\n=== PBN OUTPUT (use as input to game.py) ===")
        print(f'[Event "BEN Autoplay"]')
        print(f'[Board "{board_number}"]')
        print(f'[Dealer "{dealer}"]')
        print(f'[Vulnerable "{vuln_str}"]')
        print(f'[Deal "N:{deal_str}"]')
        print(f'[Declarer "{["N","E","S","W"][decl_i]}"]')
        print(f'[Contract "{contract}"]')
        print(f'[Result "{decl_tricks}"]')
        # Auction line - format bids in groups of 4
        auction_line = ' '.join(auction_output)
        print(f'[Auction "{dealer}"]')
        print(auction_line)
        # Play line - format cards
        opening_leader_nesw = ["N","E","S","W"][(decl_i + 1) % 4]
        print(f'[Play "{opening_leader_nesw}"]')
        # Format play in groups of 4 (one trick per line)
        for i in range(0, len(play_cards), 4):
            trick_cards = play_cards[i:i+4]
            print(' '.join(trick_cards))
        print("=== END PBN ===\n")

        # Print timing summary for this request
        print(ModelTimer.get_summary())

        return jsonify(result)

    except Exception as e:
        handle_exception(e)
        error_message = "An error occurred: {}".format(str(e))
        return jsonify({"error": error_message}), 400

@app.route('/robots.txt')
def robots_txt():
    content = "User-agent: *\nDisallow: /"
    return Response(content, mimetype='text/plain')

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