import traceback
import util
import sys
import os
import math
from objects import Card
import time
from binary import get_hcp, calculate_median
import scoring
import calculate
from bidding import bidding
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Calculate the parent directory
parent_dir = os.path.join(script_dir, "../..")
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from colorama import Fore, Back, Style, init

BEN_HOME = os.getenv('BEN_HOME') or '..'
if BEN_HOME == '.':
    BIN_FOLDER = 'bin'
else:
    BIN_FOLDER = os.path.join(BEN_HOME, 'bin')
if sys.platform == 'win32':
    BGADLL_LIB = 'BGADLL'
elif sys.platform == 'darwin':
    BGADLL_LIB = 'N/A'
else:
    BGADLL_LIB = 'N/A'

BGADLL_PATH = os.path.join(BIN_FOLDER, BGADLL_LIB)


class BGADefDLL:

    def __init__(self, models, northhand, southhand, contract, is_decl_vuln, player_i, sampler, verbose):
        try:
            # Load the .NET assembly and import the types and classes from the assembly
            util.load_dotnet_framework_assembly(BGADLL_PATH, verbose)
            from BGADLL import PIMCDef, Hand, Constraints, Extensions, Play

        except Exception as ex:
            # Provide a message to the user if the assembly is not found
            print(f"{Fore.RED}Error:", ex)
            print("*****************************************************************************")
            print("Error: Unable to load BGADLL.dll. Make sure the DLL is in the ./bin directory")
            print("Make sure the dll is not blocked by OS (Select properties and click unblock)")
            print("Make sure the dll is not write protected")
            print(f"*****************************************************************************{Fore.RESET}")
            sys.exit(1)
        if models == None:   
            return

        self.models = models       
        self.sampler = sampler     
        self.max_playout = models.pimc_max_playouts
        self.wait = models.pimc_wait
        self.autoplay = models.autoplaysingleton
        self.pimc = PIMCDef(models.pimc_max_threads, self.models.pimc_verbose and verbose)
        self.pimc.Clear()
        self.full_deck = Extensions.Parse("AKQJT98765432.AKQJT98765432.AKQJT98765432.AKQJT98765432")
        self.dummyhand = Extensions.Parse(northhand)
        self.defendinghand = Extensions.Parse(southhand)
        self.declarerhand = Hand()
        self.partnerhand = Hand()
        self.opposHand = self.full_deck.Except(self.dummyhand.Union(self.defendinghand))
        self.current_trick = Play()
        self.previous_tricks = Play()
        # Constraint are Clubs, Diamonds ending with hcp
        self.partner_constraints = Constraints(0, 13, 0, 13, 0, 13, 0, 13, 0, 37)
        self.declarer_constraints = Constraints(0, 13, 0, 13, 0, 13, 0, 13, 0, 37)
        self.already_shown_partner = [0,0,0,0]
        self.already_shown_declarer = [0,0,0,0]
        self.already_shown_hcp_partner = 0
        self.already_shown_hcp_declarer = 0
        self.suit = bidding.get_strain_i(contract)
        self.mintricks = 13 - (int(contract[0]) + 6) + 1
        self.contract = contract
        self.tricks_taken = 0
        self.score_by_tricks_taken = [scoring.score(self.contract, is_decl_vuln, n_tricks) for n_tricks in range(14)]
        self.player_i = player_i
        self.constraints_updated = False
        self.verbose = verbose

    def calculate_hcp(self, rank):
        hcp_values = {
            0: 4,
            1: 3,
            2: 2,
            3: 1,
        }
        return hcp_values.get(rank, 0)

    def reset_trick(self):
        from BGADLL import Play
        self.previous_tricks.AddRange(self.current_trick)
        self.current_trick = Play()

    def update_trick_needed(self):
        self.mintricks += -1
        self.tricks_taken += 1
        if self.verbose:
            print("mintricks",self.mintricks)
            print("tricks_taken",self.tricks_taken)
            
    def set_shape_constraints(self, min_partner, max_partner, min_declarer, max_declarer, quality):

        # Perhaps we should add a constraint on max shape for a passed hand
        # Opening lead might set some constraints on length in suit
        # ie leading an unsupported king in an unbid suit is normally single or double
        if self.constraints_updated:
            if not self.models.pimc_constraints_each_trick:
                print(f"{Fore.RED}Constraints already set{Fore.RESET}")
                return

        # Perhaps we should have a larger margin, depending on the bidding from this hand
        # if no bids, the hand can have a very long suits without having bid
        # Perhaps most important for partners hand
        if quality >= self.sampler.bidding_threshold_sampling :
            margin_declarer = self.models.pimc_margin_suit
            margin_partner = self.models.pimc_margin_suit
        else:
            margin_declarer = self.models.pimc_margin_suit_bad_samples
            margin_partner = self.models.pimc_margin_suit_bad_samples

        if self.verbose:
            print("already_shown_declarer", self.already_shown_declarer)
            print("already_shown_partner", self.already_shown_partner)

        for i in range(4):
            # If samples show 5-card+ we only reduce by 1 
            if min_partner[i] >= 5:
                min_partner[i] = max(min_partner[i] - margin_partner - self.already_shown_partner[i], 0)
            else: 
                min_partner[i] = max(min_partner[i] - margin_partner - self.already_shown_partner[i], 0)
            # If samples show 2-card- we only increase by 1 
            if max_partner[i] <= 2:
                max_partner[i] = min(max_partner[i] + margin_partner - self.already_shown_partner[i], 13)
            else: 
                max_partner[i] = min(max_partner[i] + margin_partner - self.already_shown_partner[i], 13)
            if min_declarer[i] >= 5:
                min_declarer[i] = max(min_declarer[i] - margin_declarer - self.already_shown_declarer[i], 0)
            else: 
                min_declarer[i] = max(min_declarer[i] - margin_declarer - self.already_shown_declarer[i], 0)
            if max_declarer[i] <= 2:
                max_declarer[i] = min(max_declarer[i] + margin_declarer - self.already_shown_declarer[i], 13)
            else: 
                max_declarer[i] = min(max_declarer[i] + margin_declarer - self.already_shown_declarer[i], 13)


        if self.verbose:
            print(min_partner, max_partner, min_declarer, max_declarer)

        self.partner_constraints.MinSpades = int(min_partner[0])
        self.partner_constraints.MinHearts = int(min_partner[1])
        self.partner_constraints.MinDiamonds = int(min_partner[2])
        self.partner_constraints.MinClubs = int(min_partner[3])
        self.partner_constraints.MaxSpades = int(max_partner[0])
        self.partner_constraints.MaxHearts = int(max_partner[1])
        self.partner_constraints.MaxDiamonds = int(max_partner[2])
        self.partner_constraints.MaxClubs = int(max_partner[3])
        self.declarer_constraints.MinSpades = int(min_declarer[0])
        self.declarer_constraints.MinHearts = int(min_declarer[1])
        self.declarer_constraints.MinDiamonds = int(min_declarer[2])
        self.declarer_constraints.MinClubs = int(min_declarer[3])
        self.declarer_constraints.MaxSpades = int(max_declarer[0])
        self.declarer_constraints.MaxHearts = int(max_declarer[1])
        self.declarer_constraints.MaxDiamonds = int(max_declarer[2])
        self.declarer_constraints.MaxClubs = int(max_declarer[3])
        self.constraints_updated = True

        if self.verbose:
            print("set_shape_constraints")
            print("Declarer",self.declarer_constraints.ToString())
            print("Partner",self.partner_constraints.ToString())

    def set_hcp_constraints(self, min_partner, max_partner, min_declarer, max_declarer, quality):
        # Perhaps we should add a constraint on max hcp for a passed hand
        if self.constraints_updated:
            return
        #  Constraints are for the remaining tricks and input if for full samples, so we subtract previous played card
        if self.verbose:
            print(min_partner, max_partner, min_declarer, max_declarer, quality)
            print("already_shown_d",self.already_shown_hcp_declarer)
            print("already_shown_p",self.already_shown_hcp_partner)
        if quality:
            margin = self.models.pimc_margin_hcp
        else:
            margin = self.models.pimc_margin_hcp_bad_samples
        self.declarer_constraints.MinHCP = max(min_declarer-margin-self.already_shown_hcp_declarer, 0)
        self.declarer_constraints.MaxHCP = min(max_declarer+margin-self.already_shown_hcp_declarer, 37)
        self.partner_constraints.MinHCP = max(min_partner-margin-self.already_shown_hcp_partner, 0)
        self.partner_constraints.MaxHCP = min(max_partner+margin-self.already_shown_hcp_partner, 37)

        if self.verbose:
            print("set_hcp_constraints")
            print("Declarer",self.declarer_constraints.ToString())
            print("Partner",self.partner_constraints.ToString())

    def update_constraints(self, playedBy, real_card):
        hcp = self.calculate_hcp(real_card.rank)
        suit = real_card.suit
        if self.player_i == 0:
            partner = 2
        else:
            partner = 0
        if (playedBy == 3):
            # Righty
            if suit == 0:
                self.declarer_constraints.MinSpades = max(0, self.declarer_constraints.MinSpades - 1)
                self.declarer_constraints.MaxSpades = max(0, self.declarer_constraints.MaxSpades - 1)
            if suit == 1:
                self.declarer_constraints.MinHearts = max(0, self.declarer_constraints.MinHearts - 1)
                self.declarer_constraints.MaxHearts = max(0, self.declarer_constraints.MaxHearts - 1)
            if suit == 2:
                self.declarer_constraints.MinDiamonds = max(0, self.declarer_constraints.MinDiamonds - 1)
                self.declarer_constraints.MaxDiamonds = max(0, self.declarer_constraints.MaxDiamonds - 1)
            if suit == 3:
                self.declarer_constraints.MinClubs = max(0, self.declarer_constraints.MinClubs - 1)
                self.declarer_constraints.MaxClubs = max(0, self.declarer_constraints.MaxClubs - 1)
            self.declarer_constraints.MinHCP = max(0, self.declarer_constraints.MinHCP - hcp)
            self.declarer_constraints.MaxHCP = max(0, self.declarer_constraints.MaxHCP - hcp)
        if (playedBy == partner):
            # Lefty
            if suit == 0:
                self.partner_constraints.MinSpades = max(0, self.partner_constraints.MinSpades - 1)
                self.partner_constraints.MaxSpades = max(0, self.partner_constraints.MaxSpades - 1)
            if suit == 1:
                self.partner_constraints.MinHearts = max(0, self.partner_constraints.MinHearts - 1)
                self.partner_constraints.MaxHearts = max(0, self.partner_constraints.MaxHearts - 1)
            if suit == 2:
                self.partner_constraints.MinDiamonds = max(0, self.partner_constraints.MinDiamonds - 1)
                self.partner_constraints.MaxDiamonds = max(0, self.partner_constraints.MaxDiamonds - 1)
            if suit == 3:
                self.partner_constraints.MinClubs = max(0, self.partner_constraints.MinClubs - 1)
                self.partner_constraints.MaxClubs = max(0, self.partner_constraints.MaxClubs - 1)
            self.partner_constraints.MinHCP = max(0, self.partner_constraints.MinHCP - hcp)
            self.partner_constraints.MaxHCP = max(0, self.partner_constraints.MaxHCP - hcp)
        if self.verbose:
            print("Declarer",self.declarer_constraints.ToString())
            print("Partner",self.partner_constraints.ToString())


    def set_card_played(self, card52, playedBy, openinglead):
        real_card = Card.from_code(card52)
        if self.verbose:
            print(f"Setting card {real_card} played by {playedBy} for PIMCDef")

        card = real_card.symbol_reversed()
        from BGADLL import Card as PIMCCard
        self.current_trick.Add(PIMCCard(card))
        self.opposHand.Remove(PIMCCard(card))
        suit = real_card.suit

        if playedBy == 1:
            self.dummyhand.Remove(PIMCCard(card))
        if playedBy == self.player_i:
            self.defendinghand.Remove(PIMCCard(card))

        if (playedBy == 0 and self.player_i == 2):
            self.partnerhand.Add(PIMCCard(card))
            self.already_shown_hcp_partner += self.calculate_hcp(real_card.rank)
            self.already_shown_partner[suit] += 1

        if (playedBy == 2 and self.player_i == 0):
            self.partnerhand.Add(PIMCCard(card))
            self.already_shown_hcp_partner += self.calculate_hcp(real_card.rank)
            self.already_shown_partner[suit] += 1

        if (playedBy == 3):
            self.declarerhand.Add(PIMCCard(card))
            self.already_shown_hcp_declarer += self.calculate_hcp(real_card.rank)
            self.already_shown_declarer[suit] += 1

        # We will update constraints with samples after the opening lead
        if not openinglead:
            self.update_constraints(playedBy, real_card)

    def find_trump(self, value):
        from BGADLL import Macros
        if value == 4:
            return Macros.Trump.Club
        elif value == 3:
            return Macros.Trump.Diamond
        elif value == 2:
            return Macros.Trump.Heart
        elif value == 1:
            return Macros.Trump.Spade
        elif value == 0:
            return Macros.Trump.No
        else:
            # Handle the case where value doesn't match any of the specified cases
            # This could be raising an exception, returning a default value, or any other appropriate action
            # For now, let's return None
            return None

    def update_missing_cards(self, missing_cards):
        # Define suits mapping
        suits = {
            0: "Spades",
            1: "Hearts",
            2: "Diamonds",
            3: "Clubs"
        }
        for i, suit in suits.items():
            value = int(missing_cards[i])
            # Update declarer constraints with minimum values
            if value < getattr(self.declarer_constraints, f"Min{suit}"):
                setattr(self.declarer_constraints, f"Min{suit}", value)
            if value < getattr(self.declarer_constraints, f"Max{suit}"):
                setattr(self.declarer_constraints, f"Max{suit}", int(value))
            # Update partner constraints with minimum values
            if value < getattr(self.partner_constraints, f"Min{suit}"):
                setattr(self.partner_constraints, f"Min{suit}", value)
            if value < getattr(self.partner_constraints, f"Max{suit}"):
                setattr(self.partner_constraints, f"Max{suit}", value)
            
    def update_voids(self,shown_out_suits):
        # Define suits mapping
        suits = {
            0: "Spades",
            1: "Hearts",
            2: "Diamonds",
            3: "Clubs"
        }

        # Convert shown_out_suits[0] and shown_out_suits[2] to sets for O(1) lookup
        shown_suits_lho = set(shown_out_suits[0])
        shown_suits_rho = set(shown_out_suits[2])

        # Iterate over all suits
        for suit_index, suit_name in suits.items():
            # Update LHO constraints based on shown_suits_0
            if suit_index in shown_suits_lho:
                setattr(self.declarer_constraints, f"Min{suit_name}", 0)
                setattr(self.declarer_constraints, f"Max{suit_name}", 0)
                setattr(self.declarer_constraints, f"Min{suit_name}", 0)
                setattr(self.declarer_constraints, f"Max{suit_name}", 0 if suit_index in shown_suits_rho else 13)
            
            # Update RHO constraints based on shown_suits_2
            elif suit_index in shown_suits_rho:
                setattr(self.partner_constraints, f"Min{suit_name}", 0)
                setattr(self.partner_constraints, f"Max{suit_name}", 0)
                setattr(self.partner_constraints, f"Min{suit_name}", 0)
                setattr(self.partner_constraints, f"Max{suit_name}", 0 if suit_index in shown_suits_lho else 13)

    # Define a Python function to find a bid
    def nextplay(self, player_i, shown_out_suits, missing_cards):
        from BGADLL import Constraints, Macros, Card as PIMCCard

        try:
            self.pimc.Clear()
        except Exception as ex:
            print('Error Clear:', ex)
            sys.exit(1)

        if player_i != self.player_i:
            raise Exception("player_i must be equal to self.player_i")
        

        
        self.update_voids(shown_out_suits)
        self.update_missing_cards(missing_cards)

        if self.models.pimc_apriori_probability:
            hands = [self.dummyhand, self.defendinghand, self.declarerhand, self.partnerhand]
        else:
            hands = [self.dummyhand, self.defendinghand]

        if self.verbose:
            print("player_i", self.player_i)
            print(self.dummyhand.ToString(), self.defendinghand.ToString())
            print(self.opposHand.ToString(), self.current_trick.ListAsString())
            print("Voids:", shown_out_suits)
            print(Macros.Player.West if player_i == 0 else Macros.Player.East)
            print("Over dummy", self.player_i == 2)
            print("Tricks taken", self.tricks_taken)
            print("min tricks",self.mintricks)
            print("Declarer",self.declarer_constraints.ToString())
            print("Partner",self.partner_constraints.ToString())
            print("Autoplay",self.autoplay)
            print("Current trick",self.current_trick.ListAsString())
            print("Previous tricks",self.previous_tricks.ListAsString())
            print("Other hands",self.declarerhand.ToString(), self.partnerhand.ToString())
            print("Strategy",self.models.pimc_use_fusion_strategy)

        try:
            card = self.pimc.SetupEvaluation(hands, self.opposHand, self.current_trick, self.previous_tricks, [self.declarer_constraints,
                                  self.partner_constraints], Macros.Player.East if player_i == 2 else Macros.Player.West, self.max_playout, self.autoplay, self.player_i == 2)
        except Exception as ex:        
            print(f"{Fore.RED}Error: {ex} {Fore.RESET}")
            print("player_i", player_i)
            print(self.dummyhand.ToString(), self.defendinghand.ToString())
            print(self.opposHand.ToString(), self.current_trick.ListAsString())
            print("Voids:", shown_out_suits)
            print(Macros.Player.West if player_i == 0 else Macros.Player.East)
            print("self.player_i",self.player_i)
            print("Over dummy", self.player_i == 2)
            print("Tricks taken", self.tricks_taken)
            print("min tricks",self.mintricks)
            print("Declarer",self.declarer_constraints.ToString())
            print("Partner",self.partner_constraints.ToString())
            print("Current trick",self.current_trick.ListAsString())
            print("Previous tricks",self.previous_tricks.ListAsString())
            print("Other hands",self.declarerhand.ToString(), self.partnerhand.ToString())
            print("Strategy",self.models.pimc_use_fusion_strategy)
            raise ex
        
        trump = self.find_trump(self.suit)
        if self.verbose:
            print("Trump:",trump)
            print("mintricks",self.mintricks)

        card_result = {}
        if self.autoplay:
            legalMoves = self.pimc.LegalMoves
            if len(legalMoves) == 1:
                card = legalMoves[0]
                card52 = Card.from_symbol(str(card)[::-1]).code()
                if self.verbose:
                    print("Playing only possible card:",card)
                card_result[card52] = (-1, -1, -1,"Forced card - no calculation")
                return card_result
            
        start_time = time.time()
        try:
            self.pimc.Evaluate(trump)
        except Exception as ex:
            print('Error BeginEvaluate:', ex)
            sys.exit(1)

        try:
            start_time = time.time()
            self.pimc.AwaitEvaluation(int(self.wait * 1000))
            if self.verbose:    
                print(f"Threads are finished after {time.time() - start_time:.2f}.")
        except Exception as ex:
            print('Error AwaitEvaluation:', ex)
            sys.exit(1)
        # Allow running threads to finalize
        time.sleep(0.05)
        if self.verbose:    
            print("max_playout",self.max_playout)
            print(f"Playouts: {self.pimc.Playouts}")
            print("Combinations:", self.pimc.Combinations)
            print("Examined:", self.pimc.Examined)

        try:
            legalMoves = self.pimc.LegalMoves
            results = {}
            weights = []
            e_tricks = {}
            making = {}
            for card in legalMoves:
                # Calculate win probability
                card52 = Card.from_symbol(str(card)[::-1]).code()
                #print(card52)
                self.pimc.Output.SortResults()
                x = self.pimc.Output.GetTricksWithWeights(card)
                output = list(x)
                count = float(len(output))
                # If we found no playout we need to reevaluate without constraints
                if count == 0:

                    print(f"{Fore.YELLOW}PIMCDef did not find any playouts. Trying without constraints{Fore.RESET} {self.pimc.LegalMovesToString}")
                    if self.verbose:
                        print(f"Max_playout: {self.max_playout} Playouts: {self.pimc.Playouts} Combinations: {self.pimc.Combinations} Examined:{self.pimc.Examined}")
                        print(self.dummyhand.ToString(), self.defendinghand.ToString(), self.opposHand.ToString(), self.current_trick.ListAsString())
                        print("Trump:",trump, "min tricks",self.mintricks)
                        print("Voids",shown_out_suits)
                        print(Macros.Player.West if player_i == 0 else Macros.Player.East)
                        print("self.player_i",self.player_i)
                        print("Over dummy", self.player_i == 2)
                        print("Declarer", self.declarer_constraints.ToString(), "Partner", self.partner_constraints.ToString())               
                        print("Other hands",self.declarerhand.ToString(), self.partnerhand.ToString())
                    if self.declarer_constraints.MaxHCP == 99:
                        print(f"{Fore.RED}Loop calling PIMCDef{Fore.RESET}")
                        sys.exit(1)
                    self.partner_constraints = Constraints(0, 13, 0, 13, 0, 13, 0, 13, 0, 99)
                    self.declarer_constraints = Constraints(0, 13, 0, 13, 0, 13, 0, 13, 0, 99)
                    card_result = self.nextplay(player_i, shown_out_suits, missing_cards)
                    # Reset max HCP so next call will work
                    self.partner_constraints.MaxHCP = 37
                    self.declarer_constraints.MaxHCP = 37
                    print("Done without constraints")
                    return card_result

                # Calculate total weight and makable weight
                total_weight = sum(entry.Item2 for entry in output)  # Accessing the second item in ValueTuple (assuming Single is float)
                #print("total_weight",total_weight)
                makable_weight = sum(entry.Item2 for entry in output if entry.Item1 >= self.mintricks)  # Summing weights where tricks >= mintricks
                #print("makable_weight",makable_weight)
                
                # Calculate probability
                making_probability = makable_weight / total_weight if total_weight > 0 else 0
                if math.isnan(making_probability):
                    making_probability = 0
                #print("probability",making_probability)
                
                # Calculate average tricks
                tricks_avg = sum(entry.Item1 * entry.Item2 for entry in output) / total_weight if total_weight > 0 else 0
                #print("tricks_avg",tricks_avg)
                if self.models.use_real_imp_or_mp:
                    # Iterate through the ValueTuple objects
                    results[card52] = []
                    for entry in output:
                        tricks = entry.Item1  # Access tricks
                        weight = entry.Item2  # Access weight
                        results[card52].append(tricks)
                        weights.append(weight)
                    making[card52] = making_probability
                    e_tricks[card52] = tricks_avg
                else:

                    # Second element is the score. We need to calculate it
                    score = (sum(self.score_by_tricks_taken[entry.Item1 + self.tricks_taken] * entry.Item2 for entry in output) / total_weight) if total_weight > 0 else 0
                    msg = f"Decl: {self.declarer_constraints.ToString()}|Partner: {self.partner_constraints.ToString()}|{self.pimc.Combinations} - {self.pimc.Examined} - {self.pimc.Playouts}"

                    card_result[card52] = (round(tricks_avg, 2), round(score), round(making_probability, 2), msg)
                    if self.verbose:
                        print(f"{count} {card52} {tricks_avg:.2f} {score:.0f} {making_probability:.2f}")

        except Exception as e:
            print('Error legalMoves:', e)
            traceback_str = traceback.format_exception(type(e), e, e.__traceback__)
            traceback_lines = "".join(traceback_str).splitlines()
            print(traceback_lines)  # This will print the last section starting with "File"
            sys.exit(1)

        if self.models.use_real_imp_or_mp:
            msg = f"Decl: {self.declarer_constraints.ToString()}|Partner: {self.partner_constraints.ToString()}|{self.pimc.Combinations} - {self.pimc.Examined} - {self.pimc.Playouts}"
            if self.verbose:
                print("Tricks")
                print("\n".join(f"{Card.from_code(int(k))}: [{', '.join(f'{x:>2}' for x in v[:10])}..." for k, v in results.items()))
                print("score_by_tricks_taken")
                print(self.score_by_tricks_taken)
            real_scores = calculate.calculate_score(results, self.tricks_taken, self.player_i, self.score_by_tricks_taken)
            if self.models.matchpoint:
                card_ev = calculate.calculate_mp_score(real_scores)
            else:
                if self.verbose:
                    print("Real scores")
                    print("\n".join(f"{Card.from_code(int(k))}: [{', '.join(f'{x:>5}' for x in v[:10])}..." for k, v in real_scores.items()))
                card_ev = calculate.calculate_imp_score(real_scores)

            card_result = {}
            for key in card_ev.keys():
                card_result[key] = (round(e_tricks[key], 2), round(card_ev[key],2), making[key], msg)
                if self.verbose:
                    print(f'{Card.from_code(key)} {round(e_tricks[key],3):0.3f} {round(card_ev[key],2):5.2f} {round(making[key],3):0.3f}')
                        

        if self.verbose:
            print(f"Returning {len(card_result)} from PIMCDef nextplay")

        return card_result
    
