"""
ACE (Adaptive Cardplay Engine) wrapper for BEN - Defender Play

This module provides an interface to the Ace library for defensive play,
which implements Information-Set MCTS with UCB selection for bridge cardplay.

Ace uses suit-rank card format (e.g., "HA" for Ace of Hearts).
Ace API: Game, Engine, GameOptions, ConstraintSet, Range, Config, Player, Contract
engine.Evaluate() returns List<Evaluation> with .Move, .Value, .Visits, .Depth
"""

import traceback
import util
import sys
import os
import logging
from threading import Lock
from objects import Card
import time
from binary import get_hcp, calculate_median
import scoring
import calculate
from collections import Counter

logger = logging.getLogger("ace_def")
if not logger.handlers:
    _log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'logs')
    os.makedirs(_log_dir, exist_ok=True)
    _fh = logging.FileHandler(os.path.join(_log_dir, 'ace.log'), encoding='utf-8')
    _fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(_fh)
    logger.setLevel(logging.INFO)
    logger.propagate = False

from bidding import bidding
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(script_dir, "../..")
sys.path.append(parent_dir)
from colorama import Fore, Back, Style, init
if "src" in script_dir and "ace" in script_dir:
    BIN_FOLDER = parent_dir + os.path.sep + 'bin'
else:
    BEN_HOME = os.getenv('BEN_HOME') or '.'
    if BEN_HOME == '.':
        BIN_FOLDER = 'bin'
    else:
        BIN_FOLDER = os.path.join(BEN_HOME, 'bin')

ACEDLL_LIB = 'Ace'
ACEDLL_PATH = os.path.join(BIN_FOLDER, ACEDLL_LIB)

class ACEDefDLL:

    _dll_loaded = None
    _lock = Lock()

    @classmethod
    def get_dll(cls, verbose = False):
        if cls._dll_loaded is None:
            with cls._lock:
                if cls._dll_loaded is None:
                    try:
                        util.load_dotnet_framework_assembly(ACEDLL_PATH, verbose)

                        from Ace import Game, Engine, GameOptions, ConstraintSet, Constraints, Range, Config, Contract, Extensions, Library
                        Player = Extensions.Player
                        Suit = Extensions.Suit

                        cls._dll_loaded = {
                            "Game": Game,
                            "Engine": Engine,
                            "GameOptions": GameOptions,
                            "ConstraintSet": ConstraintSet,
                            "Constraints": Constraints,
                            "Range": Range,
                            "Config": Config,
                            "Contract": Contract,
                            "Player": Player,
                            "Suit": Suit,
                            "Library": Library,
                        }

                    except Exception as ex:
                        print(f"{Fore.RED}Error: {ex}")
                        print("*****************************************************************************")
                        print("Error: Unable to load Ace.dll. Make sure the DLL is in the ./bin directory")
                        print("Make sure the dll is not blocked by OS (Select properties and click unblock)")
                        print("Make sure the dll is not write protected")
                        print(f"*****************************************************************************{Fore.RESET}")
                        traceback.print_exc()
                        sys.exit(1)
        return cls._dll_loaded

    def __init__(self, models, northhand, southhand, contract, is_decl_vuln, player_i, sampler, verbose):
        dll = ACEDefDLL.get_dll(verbose)
        if dll is None:
            raise RuntimeError("Failed to load ACE DLL. Please ensure it is properly initialized.")
        if models is None:
            return

        self.GameOptions = dll["GameOptions"]
        self.Game = dll["Game"]
        self.Engine = dll["Engine"]
        self.ConstraintSet = dll["ConstraintSet"]
        self.Constraints = dll["Constraints"]
        self.Range = dll["Range"]
        self.AceConfig = dll["Config"]
        self.Contract = dll["Contract"]
        self.Player = dll["Player"]
        self.Suit = dll["Suit"]

        self.models = models
        self.sampler = sampler
        self.verbose = verbose or getattr(models, 'ace_verbose', False)
        self.player_i = player_i

        # ACE configuration
        self.search_duration = getattr(models, 'ace_search_duration', 5000)
        self.iterations = getattr(models, 'ace_iterations', 0)
        self.search_depth = getattr(models, 'ace_search_depth', 52)
        self.search_threads = getattr(models, 'ace_threads', 10)
        self.autoplay = models.autoplaysingleton
        self.exploration = getattr(models, 'ace_exploration', 0.6061)
        self.limiter = getattr(models, 'ace_limiter', False)

        self.dummy_hand_str = northhand
        self.defender_hand_str = southhand

        self.suit = bidding.get_strain_i(contract)
        self.mintricks = 13 - (int(contract[0]) + 6) + 1
        self.contract_str = contract
        self.tricks_taken = 0
        self.score_by_tricks_taken = [scoring.score(self.contract_str, is_decl_vuln, n_tricks) for n_tricks in range(14)]

        self.played_cards = []
        self.cards_in_trick = 0

        self.already_shown_declarer = [0, 0, 0, 0]
        self.already_shown_partner = [0, 0, 0, 0]
        self.already_shown_hcp_declarer = 0
        self.already_shown_hcp_partner = 0
        self.constraints_updated = False

        self.declarer_constraints = [0, 13, 0, 13, 0, 13, 0, 13, 0, 37]
        self.partner_constraints = [0, 13, 0, 13, 0, 13, 0, 13, 0, 37]

        self.shown_voids_declarer = set()
        self.shown_voids_partner = set()

    def version(self):
        try:
            dll = ACEDefDLL.get_dll(False)
            if dll and "Library" in dll:
                return dll['Library'].Version
        except:
            pass
        return "Ace 1.0 (fallback)"

    def calculate_hcp(self, rank):
        hcp_values = {0: 4, 1: 3, 2: 2, 3: 1}
        return hcp_values.get(rank, 0)

    def reset_trick(self):
        self.cards_in_trick = 0

    def update_trick_needed(self):
        self.mintricks -= 1
        self.tricks_taken += 1
        if self.verbose:
            print("mintricks", self.mintricks)
            print("tricks_taken", self.tricks_taken)

    def set_shape_constraints(self, min_declarer, max_declarer, min_partner, max_partner, quality):
        if self.constraints_updated:
            if not self.models.pimc_constraints_each_trick:
                print(f"{Fore.RED}Constraints already set{Fore.RESET}")
                return

        if quality >= self.sampler.bid_accept_threshold_bidding:
            margin = self.models.pimc_margin_suit
        else:
            margin = self.models.pimc_margin_suit_bad_samples

        if self.verbose:
            print("already_shown_declarer", self.already_shown_declarer)
            print("already_shown_partner", self.already_shown_partner)

        trump_suit = self._get_trump_suit_index()

        for i in range(4):
            if i == trump_suit:
                continue
            if min_declarer[i] >= 5:
                min_declarer[i] = max(min_declarer[i] - 1 - self.already_shown_declarer[i], 0)
            else:
                min_declarer[i] = max(min_declarer[i] - margin - self.already_shown_declarer[i], 0)
            if max_declarer[i] <= 2:
                max_declarer[i] = min(max_declarer[i] + 1 - self.already_shown_declarer[i], 13)
            else:
                max_declarer[i] = min(max_declarer[i] + margin - self.already_shown_declarer[i], 13)
            if min_partner[i] >= 5:
                min_partner[i] = max(min_partner[i] - 1 - self.already_shown_partner[i], 0)
            else:
                min_partner[i] = max(min_partner[i] - margin - self.already_shown_partner[i], 0)
            if max_partner[i] <= 2:
                max_partner[i] = min(max_partner[i] + 1 - self.already_shown_partner[i], 13)
            else:
                max_partner[i] = min(max_partner[i] + margin - self.already_shown_partner[i], 13)

        self.declarer_constraints[0] = int(min_declarer[0])
        self.declarer_constraints[1] = int(max_declarer[0])
        self.declarer_constraints[2] = int(min_declarer[1])
        self.declarer_constraints[3] = int(max_declarer[1])
        self.declarer_constraints[4] = int(min_declarer[2])
        self.declarer_constraints[5] = int(max_declarer[2])
        self.declarer_constraints[6] = int(min_declarer[3])
        self.declarer_constraints[7] = int(max_declarer[3])

        self.partner_constraints[0] = int(min_partner[0])
        self.partner_constraints[1] = int(max_partner[0])
        self.partner_constraints[2] = int(min_partner[1])
        self.partner_constraints[3] = int(max_partner[1])
        self.partner_constraints[4] = int(min_partner[2])
        self.partner_constraints[5] = int(max_partner[2])
        self.partner_constraints[6] = int(min_partner[3])
        self.partner_constraints[7] = int(max_partner[3])

        self.constraints_updated = True

        if self.verbose:
            print("set_shape_constraints")
            print("Declarer", self.declarer_constraints)
            print("Partner", self.partner_constraints)

    def set_hcp_constraints(self, min_declarer, max_declarer, min_partner, max_partner, quality):
        if self.constraints_updated:
            return

        if quality:
            margin = self.models.pimc_margin_hcp
        else:
            margin = self.models.pimc_margin_hcp_bad_samples

        self.declarer_constraints[8] = max(min_declarer - margin - self.already_shown_hcp_declarer, 0)
        self.declarer_constraints[9] = min(max_declarer + margin - self.already_shown_hcp_declarer, 37)
        self.partner_constraints[8] = max(min_partner - margin - self.already_shown_hcp_partner, 0)
        self.partner_constraints[9] = min(max_partner + margin - self.already_shown_hcp_partner, 37)

        if self.verbose:
            print("set_hcp_constraints")
            print("Declarer", self.declarer_constraints)
            print("Partner", self.partner_constraints)

    def update_constraints(self, playedBy, real_card):
        hcp = self.calculate_hcp(real_card.rank)
        suit = real_card.suit

        if self.player_i == 0:
            if playedBy == 3:
                idx = suit * 2
                self.declarer_constraints[idx] = max(0, self.declarer_constraints[idx] - 1)
                self.declarer_constraints[idx + 1] = max(0, self.declarer_constraints[idx + 1] - 1)
                self.declarer_constraints[8] = max(0, self.declarer_constraints[8] - hcp)
                self.declarer_constraints[9] = max(0, self.declarer_constraints[9] - hcp)
            elif playedBy == 2:
                idx = suit * 2
                self.partner_constraints[idx] = max(0, self.partner_constraints[idx] - 1)
                self.partner_constraints[idx + 1] = max(0, self.partner_constraints[idx + 1] - 1)
                self.partner_constraints[8] = max(0, self.partner_constraints[8] - hcp)
                self.partner_constraints[9] = max(0, self.partner_constraints[9] - hcp)
        else:
            if playedBy == 3:
                idx = suit * 2
                self.declarer_constraints[idx] = max(0, self.declarer_constraints[idx] - 1)
                self.declarer_constraints[idx + 1] = max(0, self.declarer_constraints[idx + 1] - 1)
                self.declarer_constraints[8] = max(0, self.declarer_constraints[8] - hcp)
                self.declarer_constraints[9] = max(0, self.declarer_constraints[9] - hcp)
            elif playedBy == 0:
                idx = suit * 2
                self.partner_constraints[idx] = max(0, self.partner_constraints[idx] - 1)
                self.partner_constraints[idx + 1] = max(0, self.partner_constraints[idx + 1] - 1)
                self.partner_constraints[8] = max(0, self.partner_constraints[8] - hcp)
                self.partner_constraints[9] = max(0, self.partner_constraints[9] - hcp)

        if self.verbose:
            print("Declarer", self.declarer_constraints)
            print("Partner", self.partner_constraints)

    def _get_trump_suit_index(self):
        if self.suit == 0:
            return None
        return self.suit - 1

    def _card_to_ace_format(self, real_card):
        suits = ['S', 'H', 'D', 'C']
        ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
        return f"{suits[real_card.suit]}{ranks[real_card.rank]}"

    def _ace_card_to_ben_code(self, ace_card_str):
        suit_map = {'S': 0, 'H': 1, 'D': 2, 'C': 3}
        rank_map = {'A': 0, 'K': 1, 'Q': 2, 'J': 3, 'T': 4,
                    '9': 5, '8': 6, '7': 7, '6': 8, '5': 9,
                    '4': 10, '3': 11, '2': 12}
        suit = suit_map.get(ace_card_str[0].upper(), 0)
        rank = rank_map.get(ace_card_str[1].upper(), 0)
        return suit * 13 + rank

    def set_card_played(self, card52, playedBy, openinglead):
        real_card = Card.from_code(card52)
        if self.verbose:
            print(f"Setting card {real_card} played by {playedBy} for ACE Defender")

        ace_card = self._card_to_ace_format(real_card)
        self.played_cards.append(ace_card)
        self.cards_in_trick += 1

        suit = real_card.suit
        if self.player_i == 0:
            if playedBy == 3:
                self.already_shown_hcp_declarer += self.calculate_hcp(real_card.rank)
                self.already_shown_declarer[suit] += 1
            elif playedBy == 2:
                self.already_shown_hcp_partner += self.calculate_hcp(real_card.rank)
                self.already_shown_partner[suit] += 1
        else:
            if playedBy == 3:
                self.already_shown_hcp_declarer += self.calculate_hcp(real_card.rank)
                self.already_shown_declarer[suit] += 1
            elif playedBy == 0:
                self.already_shown_hcp_partner += self.calculate_hcp(real_card.rank)
                self.already_shown_partner[suit] += 1

        if not openinglead:
            self.update_constraints(playedBy, real_card)

    def update_missing_cards(self, missing_cards):
        for i in range(4):
            value = int(missing_cards[i])
            idx = i * 2
            if value < self.declarer_constraints[idx]:
                self.declarer_constraints[idx] = value
            if value < self.declarer_constraints[idx + 1]:
                self.declarer_constraints[idx + 1] = value
            if value < self.partner_constraints[idx]:
                self.partner_constraints[idx] = value
            if value < self.partner_constraints[idx + 1]:
                self.partner_constraints[idx + 1] = value

    def update_voids(self, shown_out_suits):
        if self.player_i == 0:
            shown_suits_declarer = set(shown_out_suits[3])
            shown_suits_partner = set(shown_out_suits[2])
        else:
            shown_suits_declarer = set(shown_out_suits[3])
            shown_suits_partner = set(shown_out_suits[0])
        for suit_index in range(4):
            idx = suit_index * 2
            if suit_index in shown_suits_declarer:
                self.declarer_constraints[idx] = 0
                self.declarer_constraints[idx + 1] = 0
                self.shown_voids_declarer.add(suit_index)
            if suit_index in shown_suits_partner:
                self.partner_constraints[idx] = 0
                self.partner_constraints[idx + 1] = 0
                self.shown_voids_partner.add(suit_index)

    def _build_pbn_deal(self):
        """Ace.dll expects 4 space-separated hands (N E S W) without seat prefix."""
        north = self.dummy_hand_str
        south = "..."
        if self.player_i == 0:
            west = self.defender_hand_str
            east = "..."
        else:
            west = "..."
            east = self.defender_hand_str
        return f"{north} {east} {south} {west}"

    def _build_constraint_set(self):
        constraint_set = self.ConstraintSet()
        south = constraint_set[self.Player.South]
        south.Spades = self.Range(self.declarer_constraints[0], self.declarer_constraints[1])
        south.Hearts = self.Range(self.declarer_constraints[2], self.declarer_constraints[3])
        south.Diamonds = self.Range(self.declarer_constraints[4], self.declarer_constraints[5])
        south.Clubs = self.Range(self.declarer_constraints[6], self.declarer_constraints[7])
        south.Hcp = self.Range(self.declarer_constraints[8], self.declarer_constraints[9])
        if self.player_i == 0:
            partner = constraint_set[self.Player.East]
        else:
            partner = constraint_set[self.Player.West]
        partner.Spades = self.Range(self.partner_constraints[0], self.partner_constraints[1])
        partner.Hearts = self.Range(self.partner_constraints[2], self.partner_constraints[3])
        partner.Diamonds = self.Range(self.partner_constraints[4], self.partner_constraints[5])
        partner.Clubs = self.Range(self.partner_constraints[6], self.partner_constraints[7])
        partner.Hcp = self.Range(self.partner_constraints[8], self.partner_constraints[9])
        return constraint_set

    def _get_ace_contract(self):
        return self.Contract.Parse(self.contract_str.replace('N', 'NT'))

    def _get_ace_declarer(self):
        return self.Player.South

    def print_dd_results(self, dd_solved, print_result=True):
        print("\n".join(f"{Card.from_code(int(k))}: [{', '.join(f'{x:>2}' for x in v[:10])}..." for k, v in dd_solved.items()))
        sorted_counts_dict = {}
        for key, array in dd_solved.items():
            element_count = Counter(array)
            sorted_counts = sorted(element_count.items(), key=lambda x: x[1], reverse=True)
            sorted_counts_dict[key] = sorted_counts
        for key, sorted_counts in sorted_counts_dict.items():
            print(f"Sorted counts for {Card.from_code(int(key))}:")
            for value, count in sorted_counts:
                print(f"  Tricks: {value}, Count: {count}")

    def _run_search_and_evaluate(self, game, depth):
        """Run ACE search and evaluate. Returns dict of card52 -> (score, display, raw, msg)."""
        engine = self.Engine.New(self.search_threads)
        engine.SetGame(game)

        if self.iterations > 0:
            engine.SetIterations(self.iterations)

        config = self.AceConfig()
        config.Exploration = self.exploration
        config.Limiter = self.limiter

        task = engine.Search(self.search_duration, self.search_duration, depth, config)
        task.Wait()

        if self.verbose:
            print(f"Search completed: {engine.Iterations} iterations in {engine.Elapsed.TotalMilliseconds:.0f}ms, depth={depth}")

        evaluations = engine.Evaluate()

        if self.verbose:
            print(f"Evaluation count: {evaluations.Count}")

        card_result = {}
        for i in range(evaluations.Count):
            ev = evaluations[i]
            reward = ev.Value
            card_str = str(ev.Move)
            card52 = self._ace_card_to_ben_code(card_str)

            elapsed_ms = engine.Elapsed.TotalMilliseconds
            msg = (f"Iterations: {engine.Iterations}"
                   f"|Win%: {reward * 100:.1f}|Visits: {ev.Visits}|Depth: {ev.Depth}"
                   f"|Time: {elapsed_ms:.0f}ms")

            card_result[card52] = (round(reward, 4), round(reward * 100), round(reward, 4), msg)

            if self.verbose:
                print(f"{Card.from_code(card52)} reward:{reward:.4f} visits:{ev.Visits}")

        return card_result

    def nextplay(self, player_i, shown_out_suits, missing_cards):
        """Find the best card to play using ACE engine for defender"""
        t_start = time.time()

        if player_i != self.player_i:
            raise Exception("player_i must be equal to self.player_i")

        self.update_voids(shown_out_suits)
        self.update_missing_cards(missing_cards)

        if self.verbose:
            print("player_i", self.player_i)
            print("Dummy:", self.dummy_hand_str)
            print("Defender:", self.defender_hand_str)
            print("Voids:", shown_out_suits)
            print("Over dummy:", self.player_i == 2)
            print("Tricks taken:", self.tricks_taken, "Tricks needed:", self.mintricks)
            print("Declarer", self.declarer_constraints)
            print("Partner", self.partner_constraints)
            print("Autoplay", self.autoplay)
            print("Played cards:", self.played_cards)

        try:
            options = self.GameOptions()
            options.Deal = self._build_pbn_deal()
            options.Declarer = self._get_ace_declarer()
            options.Contract = self._get_ace_contract()
            options.Constraints = self._build_constraint_set()

            if self.verbose:
                print("PBN Deal:", options.Deal)
                print("Contract:", self.contract_str)

            logger.info("ACE Def Request: PBN=%s Contract=%s Player=%s Played=%s Constraints=[Decl:%s Partner:%s] Depth=%s",
                        options.Deal, self.contract_str, self.player_i, self.played_cards,
                        self.declarer_constraints, self.partner_constraints,
                        self.search_depth)

            game = self.Game.New(options)

            try:
                for card_str in self.played_cards:
                    if not game.Play(card_str, False):
                        if self.verbose:
                            print(f"Warning: Could not replay card {card_str}")

                legal_moves = game.GetMoves()

                if self.verbose:
                    print(f"Legal moves: {[str(m) for m in legal_moves]}")

                if len(legal_moves) == 1 and self.autoplay:
                    card = legal_moves[0]
                    card_str = str(card)
                    card52 = self._ace_card_to_ben_code(card_str)
                    if self.verbose:
                        print(f"Playing only possible card: {card_str}")
                    return {card52: (-1, -1, -1, "Forced card - no calculation")}

                card_result = self._run_search_and_evaluate(game, self.search_depth)

                # If no results (constraints too tight), retry without constraints
                if len(card_result) == 0:
                    if self.verbose:
                        print("No results with constraints, retrying without constraints...")
                    game.Dispose()
                    options.Constraints = self.ConstraintSet()
                    game = self.Game.New(options)
                    for card_str in self.played_cards:
                        game.Play(card_str, False)
                    card_result = self._run_search_and_evaluate(game, self.search_depth)

                elapsed = time.time() - t_start
                logger.info("ACE Def Response: %s Time=%.1fs", card_result, elapsed)

            finally:
                if game is not None:
                    try:
                        game.Dispose()
                    except:
                        pass

        except Exception as e:
            logger.error("ACE Def nextplay error: %s", e, exc_info=True)
            print(f'{Fore.RED}Error in ACE Def nextplay: {e}{Fore.RESET}')
            traceback.print_exc()
            return {}

        if self.verbose:
            print(f"Returning {len(card_result)} from ACE Def nextplay")
            print(f'ACE response time: {time.time() - t_start:0.4f}')

        return card_result
