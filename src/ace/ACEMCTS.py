"""
ACE-MCTS (Monte Carlo Tree Search) wrapper for BEN - Declarer Play

This module provides an interface to the Brill.Ace.Mcts library, which implements
Information-Set MCTS with UCB selection for bridge cardplay.

Brill.Ace.Mcts uses suit-rank card format (e.g., "HA" for Ace of Hearts).
Brill.Ace.Mcts API: Game, Engine, GameOptions, ConstraintSet, Range, Config, Player, Contract
Key difference from ACE: No Model class. Uses Config with Confidence/Exploration parameters.
engine.Evaluate() returns List<Evaluation> with .Move, .Value, .Visits, .Depth
"""

import traceback
import util
import sys
import os
import logging
from threading import Lock
import math
from objects import Card
import time
from binary import get_hcp, calculate_median
import scoring
import calculate
from collections import Counter
import asyncio

logger = logging.getLogger("ace_mcts")
if not logger.handlers:
    _log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'logs')
    os.makedirs(_log_dir, exist_ok=True)
    _fh = logging.FileHandler(os.path.join(_log_dir, 'ace_mcts.log'), encoding='utf-8')
    _fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(_fh)
    logger.setLevel(logging.INFO)
    logger.propagate = False

from bidding import bidding
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Calculate the parent directory
parent_dir = os.path.join(script_dir, "../..")
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from colorama import Fore, Back, Style, init
if "src" in script_dir and "ace" in script_dir:
    # We are running inside the src/ace directory
    BIN_FOLDER = parent_dir + os.path.sep + 'bin'
else:

    BEN_HOME = os.getenv('BEN_HOME') or '.'
    if BEN_HOME == '.':
        BIN_FOLDER = 'bin'
    else:
        BIN_FOLDER = os.path.join(BEN_HOME, 'bin')

if sys.platform == 'win32':
    MCTSDLL_LIB = 'Brill.Ace.Mcts'
elif sys.platform == 'darwin':
    MCTSDLL_LIB = 'N/A'
else:
    MCTSDLL_LIB = 'N/A'

MCTSDLL_PATH = os.path.join(BIN_FOLDER, MCTSDLL_LIB)

class ACEMCTSDLL:

    _dll_loaded = None  # Class-level attribute to store the DLL singleton
    _lock = Lock()      # Lock to ensure thread-safe initialization

    @classmethod
    def get_dll(cls, verbose = False):
        """Access the loaded DLL classes."""
        if cls._dll_loaded is None:
            with cls._lock:  # Ensure only one thread can enter this block at a time
                if cls._dll_loaded is None:  # Double-checked locking
                    try:
                        # Load the .NET assembly and import the types and classes from the assembly
                        util.load_dotnet_framework_assembly(MCTSDLL_PATH, verbose)

                        from Brill.Ace.Mcts import Game, Engine, GameOptions, ConstraintSet, Constraints, Range, Config, Contract, Extensions, BcalcddsSolver, HaglundSolver, Evaluation
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
                            "BcalcddsSolver": BcalcddsSolver,
                            "HaglundSolver": HaglundSolver,
                            "Evaluation": Evaluation
                        }

                    except Exception as ex:
                        # Provide a message to the user if the assembly is not found
                        print(f"{Fore.RED}Error: {ex}")
                        print("*****************************************************************************")
                        print("Error: Unable to load Brill.Ace.Mcts.dll. Make sure the DLL is in the ./bin directory")
                        print("Make sure the dll is not blocked by OS (Select properties and click unblock)")
                        print("Make sure the dll is not write protected")
                        print(f"*****************************************************************************{Fore.RESET}")
                        traceback.print_exc()
                        sys.exit(1)
        return cls._dll_loaded

    def __init__(self, models, northhand, southhand, contract, is_decl_vuln, sampler, verbose):
        dll = ACEMCTSDLL.get_dll(verbose)  # Retrieve the loaded DLL classes through the singleton
        if dll is None:
            raise RuntimeError("Failed to load ACE-MCTS DLL. Please ensure it is properly initialized.")
        if models is None:
            return

        # Access classes from the DLL
        self.GameOptions = dll["GameOptions"]
        self.Game = dll["Game"]
        self.Engine = dll["Engine"]
        self.ConstraintSet = dll["ConstraintSet"]
        self.Constraints = dll["Constraints"]
        self.Range = dll["Range"]
        self.MctsConfig = dll["Config"]
        self.Contract = dll["Contract"]
        self.Player = dll["Player"]
        self.Suit = dll["Suit"]

        self.models = models
        self.sampler = sampler
        self.verbose = verbose or getattr(models, 'ace_mcts_verbose', False)

        # ACE-MCTS configuration
        self.search_duration = getattr(models, 'ace_mcts_search_duration', 2000)  # ms
        self.max_iterations = getattr(models, 'ace_mcts_max_iterations', 0)  # 0 = unlimited
        self.max_passed_samples = getattr(models, 'ace_mcts_max_passed_samples', 200)
        self.search_depth = getattr(models, 'ace_mcts_search_depth', 2)
        self.search_threads = getattr(models, 'ace_mcts_threads', 10)
        self.autoplay = models.autoplaysingleton
        # MCTS-specific parameters (no model concept)
        self.confidence = getattr(models, 'ace_mcts_confidence', 0.9004)
        self.exploration = getattr(models, 'ace_mcts_exploration', 0.6061)

        # DDS library selection: 'bcalcdds' (default) or 'haglund' (Bo Haglund's dds.dll)
        self.dds_library = getattr(models, 'ace_mcts_dds_library', 'bcalcdds').lower()

        # Create the solver based on config
        self.BcalcddsSolver = dll["BcalcddsSolver"]
        self.HaglundSolver = dll["HaglundSolver"]

        if self.dds_library == 'haglund':
            if self.HaglundSolver.IsAvailable:
                self.solver = self.HaglundSolver()
                if verbose:
                    print(f"ACE-MCTS using Bo Haglund's dds.dll")
            else:
                self.solver = self.BcalcddsSolver()
                if verbose:
                    print(f"ACE-MCTS: dds.dll not available, falling back to libbcalcdds")
        else:
            self.solver = self.BcalcddsSolver()
            if verbose:
                print(f"ACE-MCTS using libbcalcdds")

        # Store hand information in PBN format for deal construction
        self.dummy_hand_str = northhand
        self.declarer_hand_str = southhand

        # Contract parsing
        self.suit = bidding.get_strain_i(contract)
        self.mintricks = int(contract[0]) + 6
        self.contract_str = contract
        self.tricks_taken = 0
        self.score_by_tricks_taken = [scoring.score(self.contract_str, is_decl_vuln, n_tricks) for n_tricks in range(14)]

        # Cards played tracking
        self.played_cards = []
        self.cards_in_trick = 0

        # Constraint tracking for opponents
        self.already_shown_rho = [0, 0, 0, 0]
        self.already_shown_lho = [0, 0, 0, 0]
        self.already_shown_hcp_rho = 0
        self.already_shown_hcp_lho = 0
        self.constraints_updated = False

        # Constraint ranges: [min_spades, max_spades, min_hearts, max_hearts, min_diamonds, max_diamonds, min_clubs, max_clubs, min_hcp, max_hcp]
        self.lho_constraints = [0, 13, 0, 13, 0, 13, 0, 13, 0, 37]
        self.rho_constraints = [0, 13, 0, 13, 0, 13, 0, 13, 0, 37]

        # Track voids shown by opponents
        self.shown_voids_lho = set()
        self.shown_voids_rho = set()

    def version(self):
        try:
            dll = ACEMCTSDLL.get_dll(False)
            if dll and "Engine" in dll:
                return dll["Engine"].Version
        except:
            pass
        return "Brill.Ace.Mcts 1.0 (fallback)"

    def calculate_hcp(self, rank):
        """Calculate HCP value for a card rank (0=A, 1=K, 2=Q, 3=J)"""
        hcp_values = {
            0: 4,
            1: 3,
            2: 2,
            3: 1,
        }
        return hcp_values.get(rank, 0)

    def reset_trick(self):
        """Reset for new trick"""
        self.cards_in_trick = 0

    def update_trick_needed(self):
        """Update after winning a trick"""
        self.mintricks -= 1
        self.tricks_taken += 1
        if self.verbose:
            print("mintricks", self.mintricks)
            print("tricks_taken", self.tricks_taken)

    def set_shape_constraints(self, min_lho, max_lho, min_rho, max_rho, quality):
        """Set shape constraints for opponents based on bidding samples"""
        if self.constraints_updated:
            if not self.models.pimc_constraints_each_trick:
                print(f"{Fore.RED}Constraints already set{Fore.RESET}")
                return

        if quality >= self.sampler.bid_accept_threshold_bidding:
            margin = self.models.pimc_margin_suit
        else:
            margin = self.models.pimc_margin_suit_bad_samples

        if self.verbose:
            print("already_shown_lho", self.already_shown_lho)
            print("already_shown_rho", self.already_shown_rho)

        trump_suit = self._get_trump_suit_index()

        for i in range(4):
            if i == trump_suit:
                continue
            if min_lho[i] >= 5:
                min_lho[i] = max(min_lho[i] - 1 - self.already_shown_lho[i], 0)
            else:
                min_lho[i] = max(min_lho[i] - margin - self.already_shown_lho[i], 0)
            if max_lho[i] <= 2:
                max_lho[i] = min(max_lho[i] + 1 - self.already_shown_lho[i], 13)
            else:
                max_lho[i] = min(max_lho[i] + margin - self.already_shown_lho[i], 13)
            if min_rho[i] >= 5:
                min_rho[i] = max(min_rho[i] - 1 - self.already_shown_rho[i], 0)
            else:
                min_rho[i] = max(min_rho[i] - margin - self.already_shown_rho[i], 0)
            if max_rho[i] <= 2:
                max_rho[i] = min(max_rho[i] + 1 - self.already_shown_rho[i], 13)
            else:
                max_rho[i] = min(max_rho[i] + margin - self.already_shown_rho[i], 13)

        self.lho_constraints[0] = int(min_lho[0])
        self.lho_constraints[1] = int(max_lho[0])
        self.lho_constraints[2] = int(min_lho[1])
        self.lho_constraints[3] = int(max_lho[1])
        self.lho_constraints[4] = int(min_lho[2])
        self.lho_constraints[5] = int(max_lho[2])
        self.lho_constraints[6] = int(min_lho[3])
        self.lho_constraints[7] = int(max_lho[3])

        self.rho_constraints[0] = int(min_rho[0])
        self.rho_constraints[1] = int(max_rho[0])
        self.rho_constraints[2] = int(min_rho[1])
        self.rho_constraints[3] = int(max_rho[1])
        self.rho_constraints[4] = int(min_rho[2])
        self.rho_constraints[5] = int(max_rho[2])
        self.rho_constraints[6] = int(min_rho[3])
        self.rho_constraints[7] = int(max_rho[3])

        self.constraints_updated = True

        if self.verbose:
            print("set_shape_constraints")
            print("East (RHO)", self.rho_constraints)
            print("West (LHO)", self.lho_constraints)

    def set_hcp_constraints(self, min_lho, max_lho, min_rho, max_rho, quality):
        """Set HCP constraints for opponents"""
        if self.constraints_updated:
            return

        if self.verbose:
            print("already_shown_lho", self.already_shown_hcp_lho)
            print("already_shown_rho", self.already_shown_hcp_rho)

        if quality:
            margin = self.models.pimc_margin_hcp
        else:
            margin = self.models.pimc_margin_hcp_bad_samples

        self.lho_constraints[8] = max(min_lho - margin - self.already_shown_hcp_lho, 0)
        self.lho_constraints[9] = min(max_lho + margin - self.already_shown_hcp_lho, 37)
        self.rho_constraints[8] = max(min_rho - margin - self.already_shown_hcp_rho, 0)
        self.rho_constraints[9] = min(max_rho + margin - self.already_shown_hcp_rho, 37)

        if self.verbose:
            print("set_hcp_constraints")
            print("East (RHO)", self.rho_constraints)
            print("West (LHO)", self.lho_constraints)

    def update_constraints(self, playedBy, real_card):
        """Update constraints after a card is played"""
        hcp = self.calculate_hcp(real_card.rank)
        suit = real_card.suit

        if playedBy == 2:  # RHO (East)
            idx = suit * 2
            self.rho_constraints[idx] = max(0, self.rho_constraints[idx] - 1)
            self.rho_constraints[idx + 1] = max(0, self.rho_constraints[idx + 1] - 1)
            self.rho_constraints[8] = max(0, self.rho_constraints[8] - hcp)
            self.rho_constraints[9] = max(0, self.rho_constraints[9] - hcp)

        if playedBy == 0:  # LHO (West)
            idx = suit * 2
            self.lho_constraints[idx] = max(0, self.lho_constraints[idx] - 1)
            self.lho_constraints[idx + 1] = max(0, self.lho_constraints[idx + 1] - 1)
            self.lho_constraints[8] = max(0, self.lho_constraints[8] - hcp)
            self.lho_constraints[9] = max(0, self.lho_constraints[9] - hcp)

        if self.verbose:
            print("East (RHO)", self.rho_constraints)
            print("West (LHO)", self.lho_constraints)

    def _get_trump_suit_index(self):
        """Get trump suit index (0=S, 1=H, 2=D, 3=C, None=NT)"""
        if self.suit == 0:
            return None
        return self.suit - 1

    def _card_to_ace_format(self, real_card):
        """Convert BEN card to Brill.Ace.Mcts suit-rank format (e.g., 'HA' for Ace of Hearts)"""
        suits = ['S', 'H', 'D', 'C']
        ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
        suit_char = suits[real_card.suit]
        rank_char = ranks[real_card.rank]
        return f"{suit_char}{rank_char}"

    def _ace_card_to_ben_code(self, ace_card_str):
        """Convert Brill.Ace.Mcts card string to BEN card code (0-51)"""
        suit_char = ace_card_str[0]
        rank_char = ace_card_str[1]

        suit_map = {'S': 0, 'H': 1, 'D': 2, 'C': 3}
        rank_map = {'A': 0, 'K': 1, 'Q': 2, 'J': 3, 'T': 4,
                    '9': 5, '8': 6, '7': 7, '6': 8, '5': 9,
                    '4': 10, '3': 11, '2': 12}

        suit = suit_map.get(suit_char.upper(), 0)
        rank = rank_map.get(rank_char.upper(), 0)

        return suit * 13 + rank

    def set_card_played(self, card52, playedBy, openinglead):
        """Record a card that has been played"""
        real_card = Card.from_code(card52)
        if self.verbose:
            print(f"Setting card {real_card} played by {playedBy} for ACE-MCTS")

        ace_card = self._card_to_ace_format(real_card)
        self.played_cards.append(ace_card)
        self.cards_in_trick += 1

        suit = real_card.suit

        if playedBy == 0:  # LHO (West)
            self.already_shown_hcp_lho += self.calculate_hcp(real_card.rank)
            self.already_shown_lho[suit] += 1
        elif playedBy == 2:  # RHO (East)
            self.already_shown_hcp_rho += self.calculate_hcp(real_card.rank)
            self.already_shown_rho[suit] += 1

        if not openinglead:
            self.update_constraints(playedBy, real_card)

    def update_missing_cards(self, missing_cards):
        """Update constraints based on cards missing from suits"""
        for i in range(4):
            value = int(missing_cards[i])
            idx = i * 2
            if value < self.lho_constraints[idx]:
                self.lho_constraints[idx] = value
            if value < self.lho_constraints[idx + 1]:
                self.lho_constraints[idx + 1] = value
            if value < self.rho_constraints[idx]:
                self.rho_constraints[idx] = value
            if value < self.rho_constraints[idx + 1]:
                self.rho_constraints[idx + 1] = value

    def update_voids(self, shown_out_suits):
        """Update constraints when a player shows out of a suit"""
        shown_suits_lho = set(shown_out_suits[0])
        shown_suits_rho = set(shown_out_suits[2])

        for suit_index in range(4):
            idx = suit_index * 2
            if suit_index in shown_suits_lho:
                self.lho_constraints[idx] = 0
                self.lho_constraints[idx + 1] = 0
                self.shown_voids_lho.add(suit_index)
            if suit_index in shown_suits_rho:
                self.rho_constraints[idx] = 0
                self.rho_constraints[idx + 1] = 0
                self.shown_voids_rho.add(suit_index)

    def _build_pbn_deal(self):
        """Build PBN deal string from current state"""
        north = self.dummy_hand_str
        south = self.declarer_hand_str
        east = "..."
        west = "..."
        return f"N:{north} {east} {south} {west}"

    def _build_constraint_set(self):
        """Build ConstraintSet from current constraints"""
        constraint_set = self.ConstraintSet()

        west = constraint_set[self.Player.West]
        west.Spades = self.Range(self.lho_constraints[0], self.lho_constraints[1])
        west.Hearts = self.Range(self.lho_constraints[2], self.lho_constraints[3])
        west.Diamonds = self.Range(self.lho_constraints[4], self.lho_constraints[5])
        west.Clubs = self.Range(self.lho_constraints[6], self.lho_constraints[7])
        west.Hcp = self.Range(self.lho_constraints[8], self.lho_constraints[9])

        east = constraint_set[self.Player.East]
        east.Spades = self.Range(self.rho_constraints[0], self.rho_constraints[1])
        east.Hearts = self.Range(self.rho_constraints[2], self.rho_constraints[3])
        east.Diamonds = self.Range(self.rho_constraints[4], self.rho_constraints[5])
        east.Clubs = self.Range(self.rho_constraints[6], self.rho_constraints[7])
        east.Hcp = self.Range(self.rho_constraints[8], self.rho_constraints[9])

        return constraint_set

    def _get_ace_contract(self):
        """Convert BEN contract to Brill.Ace.Mcts Contract"""
        return self.Contract.Parse(self.contract_str.replace('N', 'NT'))

    def _get_ace_declarer(self):
        """Get declarer position (South for declarer play)"""
        return self.Player.South

    def print_dd_results(self, dd_solved, print_result=True):
        """Print detailed results (same as PIMC for compatibility)"""
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

    def nextplay(self, player_i, shown_out_suits, missing_cards):
        """Find the best card to play using ACE-MCTS engine"""
        t_start = time.time()

        self.update_voids(shown_out_suits)
        self.update_missing_cards(missing_cards)

        if self.verbose:
            print("player_i", player_i)
            print("Dummy:", self.dummy_hand_str)
            print("Declarer:", self.declarer_hand_str)
            print("Voids:", shown_out_suits)
            print("Tricks taken:", self.tricks_taken, "Tricks needed:", self.mintricks)
            print("East (RHO)", self.rho_constraints)
            print("West (LHO)", self.lho_constraints)
            print("Autoplay", self.autoplay)
            print("Played cards:", self.played_cards)

        try:
            # Build game options
            options = self.GameOptions()
            options.Deal = self._build_pbn_deal()
            options.Declarer = self._get_ace_declarer()
            options.Contract = self._get_ace_contract()
            options.Constraints = self._build_constraint_set()

            if self.verbose:
                print("PBN Deal:", options.Deal)
                print("Contract:", self.contract_str)

            logger.info("ACE-MCTS Request: PBN=%s Contract=%s Played=%s Constraints=[RHO:%s LHO:%s] Depth=%s Confidence=%s Exploration=%s",
                        options.Deal, self.contract_str, self.played_cards,
                        self.rho_constraints, self.lho_constraints,
                        self.search_depth, self.confidence, self.exploration)

            # Create game and replay played cards
            game = self.Game.New(options)

            try:
                # Replay all previously played cards
                for card_str in self.played_cards:
                    if not game.Play(card_str, False):
                        if self.verbose:
                            print(f"Warning: Could not replay card {card_str}")

                # Get legal moves
                legal_moves = game.GetMoves()

                if self.verbose:
                    print(f"Legal moves: {[str(m) for m in legal_moves]}")

                # If only one legal move, return it immediately
                if len(legal_moves) == 1 and self.autoplay:
                    card = legal_moves[0]
                    card_str = str(card)
                    card52 = self._ace_card_to_ben_code(card_str)
                    if self.verbose:
                        print(f"Playing only possible card: {card_str}")
                    return {card52: (-1, -1, -1, "Forced card - no calculation")}

                # Create engine and run search with the configured DDS solver
                engine = self.Engine.New(self.search_threads, self.solver)
                engine.SetGame(game)

                # Set limits if configured (0 = unlimited)
                if self.max_iterations > 0:
                    engine.SetMaxIterations(self.max_iterations)
                if self.max_passed_samples > 0:
                    engine.SetMaxPassedSamples(self.max_passed_samples)

                # Calculate adaptive depth (capped by configured search_depth)
                adaptive_depth = max(1, min(self.search_depth, self.cards_in_trick + 1))

                # Build MCTS Config with Confidence and Exploration
                mcts_config = self.MctsConfig()
                mcts_config.Confidence = self.confidence
                mcts_config.Exploration = self.exploration

                # Run search with MCTS config
                task = engine.Search(self.search_duration, self.search_duration, adaptive_depth, mcts_config)
                task.Wait()

                if self.verbose:
                    print(f"MCTS Search completed: {engine.Iterations} iterations, {engine.PassedSamples} passed in {engine.Elapsed.TotalMilliseconds:.0f}ms, depth={adaptive_depth}")
                    print(f"Final relaxation level: {engine.FinalRelaxationLevel}")
                    print(f"Config: confidence={self.confidence}, exploration={self.exploration}")

                # MCTS Evaluate returns List<Evaluation> sorted best-first
                # Each Evaluation has .Move, .Value (win rate 0-1), .Visits, .Depth
                evaluations = engine.Evaluate()

                if self.verbose:
                    print("MCTS Evaluation results:")
                    for i in range(evaluations.Count):
                        ev = evaluations[i]
                        print(f"  {ev.Move}: reward={ev.Value:.4f} visits={ev.Visits} depth={ev.Depth}")

                # Convert results to BEN format
                card_result = {}
                for i in range(evaluations.Count):
                    ev = evaluations[i]
                    reward = ev.Value  # Win rate 0-1, higher = better
                    card_str = str(ev.Move)
                    card52 = self._ace_card_to_ben_code(card_str)

                    elapsed_ms = engine.Elapsed.TotalMilliseconds
                    msg = (f"Iterations: {engine.Iterations}|Samples: {engine.PassedSamples}"
                           f"|Win%: {reward * 100:.1f}|Visits: {ev.Visits}|Depth: {ev.Depth}"
                           f"|Time: {elapsed_ms:.0f}ms")
                    if engine.FinalRelaxationLevel > 0:
                        msg += f"|Relaxation: {engine.FinalRelaxationLevel}"

                    # Tuple: (ranking_score, display_score, raw_score, message)
                    card_result[card52] = (round(reward, 4), round(reward * 100), round(reward, 4), msg)

                    if self.verbose:
                        print(f"{Card.from_code(card52)} reward:{reward:.4f} visits:{ev.Visits}")

                elapsed = time.time() - t_start
                evals_str = " ".join(f"{str(evaluations[i].Move)}:{evaluations[i].Value:.4f}({evaluations[i].Visits})" for i in range(evaluations.Count))
                logger.info("ACE-MCTS Response: %s Iterations=%s Passed=%s Depth=%s Relaxation=%s Time=%.1fs",
                            evals_str, engine.Iterations, engine.PassedSamples, adaptive_depth,
                            engine.FinalRelaxationLevel, elapsed)

            finally:
                if game is not None:
                    try:
                        game.Dispose()
                    except:
                        pass

        except Exception as e:
            logger.error("ACE-MCTS nextplay error: %s", e, exc_info=True)
            print(f'{Fore.RED}Error in ACE-MCTS nextplay: {e}{Fore.RESET}')
            traceback.print_exc()
            return {}

        if self.verbose:
            print(f"Returning {len(card_result)} from ACE-MCTS nextplay")
            print(f'ACE-MCTS response time: {time.time() - t_start:0.4f}')

        return card_result
