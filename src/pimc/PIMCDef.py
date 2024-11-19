import traceback
import clr
import sys
import os
import math
from objects import Card
import time
from binary import get_hcp, calculate_median
import scoring
import calculate
from bidding import bidding
sys.path.append("..")
from colorama import Fore, Back, Style, init

BEN_HOME = os.getenv('BEN_HOME') or '..'
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
            try:
                clr.AddReference(BGADLL_PATH)
            except Exception as ex:
                # Try loading with .dll extension
                clr.AddReference(BGADLL_PATH+'.dll')
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
        if self.constraints_updated:
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

        # Perhaps we should use another margin for trump

        allready_shown_partner = [0,0,0,0]
        allready_shown_partner[0] = 13 - self.partner_constraints.MaxSpades
        allready_shown_partner[1] = 13 - self.partner_constraints.MaxHearts
        allready_shown_partner[2] = 13 - self.partner_constraints.MaxDiamonds
        allready_shown_partner[3] = 13 - self.partner_constraints.MaxClubs
        allready_shown_declarer = [0,0,0,0]
        allready_shown_declarer[0] = 13 - self.declarer_constraints.MaxSpades
        allready_shown_declarer[1] = 13 - self.declarer_constraints.MaxHearts
        allready_shown_declarer[2] = 13 - self.declarer_constraints.MaxDiamonds
        allready_shown_declarer[3] = 13 - self.declarer_constraints.MaxClubs

        if self.verbose:
            print("allready_shown_declarer",allready_shown_declarer)
            print("allready_shown_partner",allready_shown_partner)

        for i in range(4):
            min_partner[i] = max(min_partner[i] - margin_partner - allready_shown_partner[i], 0)
            max_partner[i] = min(max_partner[i] + margin_partner - allready_shown_partner[i], 13)
            min_declarer[i] = max(min_declarer[i] - margin_declarer - allready_shown_declarer[i], 0)
            max_declarer[i] = min(max_declarer[i] + margin_declarer - allready_shown_declarer[i], 13)

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
        allready_shown_declarer = 37 - self.declarer_constraints.MaxHCP
        allready_shown_partner = 37 - self.partner_constraints.MaxHCP
        if self.verbose:
            print(min_partner, max_partner, min_declarer, max_declarer, quality)
            print("allready_shown_d",allready_shown_declarer)
            print("allready_shown_p",allready_shown_partner)
        if quality:
            margin = self.models.pimc_margin_hcp
        else:
            margin = self.models.pimc_margin_hcp_bad_samples
        self.declarer_constraints.MinHCP = max(min_declarer-margin-allready_shown_declarer, 0)
        self.declarer_constraints.MaxHCP = min(max_declarer+margin-allready_shown_declarer, 37)
        self.partner_constraints.MinHCP = max(min_partner-margin-allready_shown_partner, 0)
        self.partner_constraints.MaxHCP = min(max_partner+margin-allready_shown_partner, 37)

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
        if playedBy == 1:
            self.dummyhand.Remove(PIMCCard(card))
        if playedBy == self.player_i:
            self.defendinghand.Remove(PIMCCard(card))
        if (playedBy == 0 and self.player_i == 2):
            self.partnerhand.Add(PIMCCard(card))
        if (playedBy == 2 and self.player_i == 0):
            self.partnerhand.Add(PIMCCard(card))
        if (playedBy == 3):
            self.declarerhand.Add(PIMCCard(card))

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

    # Define a Python function to find a bid
    def nextplay(self, player_i, shown_out_suits):
        from BGADLL import Constraints, Macros, Card as PIMCCard

        try:
            self.pimc.Clear()
        except Exception as ex:
            print('Error Clear:', ex)
            sys.exit(1)

        if player_i != self.player_i:
            raise Exception("player_i must be equal to self.player_i")
        

        # Declarer
        idx = 3
        # Partner
        if self.player_i == 0:
            idx1 = 2
        else:
            idx1 = 0
        if 0 in shown_out_suits[idx]:
            self.declarer_constraints.MinSpades = 0
            self.declarer_constraints.MaxSpades = 0
            self.partner_constraints.MinSpades = 0
            self.partner_constraints.MaxSpades = 13
        if 1 in shown_out_suits[idx]:
            self.declarer_constraints.MinHearts = 0
            self.declarer_constraints.MaxHearts = 0
            self.partner_constraints.MinHearts = 0
            self.partner_constraints.MaxHearts = 13
        if 2 in shown_out_suits[idx]:
            self.declarer_constraints.MinDiamonds = 0
            self.declarer_constraints.MaxDiamonds = 0
            self.partner_constraints.MinDiamonds = 0
            self.partner_constraints.MaxDiamonds = 13
        if 3 in shown_out_suits[idx]:
            self.declarer_constraints.MinClubs = 0
            self.declarer_constraints.MaxClubs = 0
            self.partner_constraints.MinClubs = 0
            self.partner_constraints.MaxClubs = 13
        if 0 in shown_out_suits[idx1]:
            self.partner_constraints.MinSpades = 0
            self.partner_constraints.MaxSpades = 0
            self.declarer_constraints.MinSpades = 0
            self.declarer_constraints.MaxSpades = 13
        if 1 in shown_out_suits[idx1]:
            self.partner_constraints.MinHearts = 0
            self.partner_constraints.MaxHearts = 0
            self.declarer_constraints.MinHearts = 0
            self.declarer_constraints.MaxHearts = 13
        if 2 in shown_out_suits[idx1]:
            self.partner_constraints.MinDiamonds = 0
            self.partner_constraints.MaxDiamonds = 0
            self.declarer_constraints.MinDiamonds = 0
            self.declarer_constraints.MaxDiamonds = 13
        if 3 in shown_out_suits[idx1]:
            self.partner_constraints.MinClubs = 0
            self.partner_constraints.MaxClubs = 0
            self.declarer_constraints.MinClubs = 0
            self.declarer_constraints.MaxClubs = 13

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
            sys.exit(1) 
        
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

                    print(card)
                    print("Trying without constraints")
                    print("max_playout",self.max_playout)
                    print(self.pimc.LegalMovesToString)
                    print(f"Playouts: {self.pimc.Playouts}")
                    print("Combinations:", self.pimc.Combinations)
                    print("Examined:", self.pimc.Examined)
                    print(self.dummyhand.ToString(), self.defendinghand.ToString())
                    print(self.opposHand.ToString(), self.current_trick.ListAsString())
                    print("min tricks",self.mintricks)
                    print("Voids",shown_out_suits)
                    print(Macros.Player.West if player_i == 0 else Macros.Player.East)
                    print("self.player_i",self.player_i)
                    print("Over dummy", self.player_i == 2)
                    print("Declarer", self.declarer_constraints.ToString())
                    print("Partner", self.partner_constraints.ToString())
                    print("Trump:",trump)
                    print("Other hands",self.declarerhand.ToString(), self.partnerhand.ToString())
                    if self.declarer_constraints.MaxHCP == 99:
                        print("Loop calling PIMC")
                        sys.exit(1)
                    self.partner_constraints = Constraints(0, 13, 0, 13, 0, 13, 0, 13, 0, 99)
                    self.declarer_constraints = Constraints(0, 13, 0, 13, 0, 13, 0, 13, 0, 99)
                    card_result = self.nextplay(player_i, shown_out_suits)
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
                    msg = f"LHO: {self.lho_constraints.ToString()}|RHO: {self.rho_constraints.ToString()}|{self.pimc.Combinations} - {self.pimc.Examined} - {self.pimc.Playouts}"

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
            real_scores = calculate.calculate_score(results, self.tricks_taken, self.player_i, self.score_by_tricks_taken)
            if self.models.matchpoint:
                card_ev = calculate.calculate_mp_score(real_scores)
            else:
                if self.verbose:
                    print("Real scores")
                    print(real_scores)
                card_ev = calculate.calculate_imp_score(real_scores)

            card_result = {}
            for key in card_ev.keys():
                card_result[key] = (round(e_tricks[key], 2), round(card_ev[key],2), making[key], msg)
                if self.verbose:
                    print(f'{key} {round(e_tricks[key],3):0.3f} {round(card_ev[key],2):5.2f} {round(making[key],3):0.3f}')
                        

        if self.verbose:
            print(card_result)
            print(f"Returning {len(card_result)} from PIMCDef nextplay")

        return card_result
    
