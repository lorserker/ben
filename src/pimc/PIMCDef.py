from typing import List, Iterable
from enum import Enum
import asyncio
import clr
import ctypes
import sys
import os
import math
from objects import Card
import time
from binary import get_hcp
import scoring

from bidding import bidding
sys.path.append("..")

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

    def __init__(self, models, northhand, southhand, contract, is_decl_vuln, player_i, verbose):
        try:
           # Load the .NET assembly and import the types and classes from the assembly
            clr.AddReference(BGADLL_PATH)
            from BGADLL import PIMCDef, Hand, Constraints, Macros, Extensions

        except Exception as ex:
            # Provide a message to the user if the assembly is not found
            print("Error: Unable to load BGADLL.dll. Make sure the DLL is in the ./bin directory")
            print("Make sure the dll is not blocked by OS (Select properties and click unblock)")
            print("Make sure the dll is not writeprotected")
            print('Error:', ex)
            raise ex
        
        self.max_playout = models.pimc_max_playout
        self.wait = models.pimc_wait
        self.autoplay = models.pimc_autoplaysingleton
        self.pimc = PIMCDef(models.pimc_max_threads)
        self.pimc.Clear()
        self.full_deck = Extensions.Parse("AKQJT98765432.AKQJT98765432.AKQJT98765432.AKQJT98765432")
        self.dummyhand = Extensions.Parse(northhand)
        self.defendinghand = Extensions.Parse(southhand)
        self.opposHand = self.full_deck.Except(self.dummyhand.Union(self.defendinghand))
        self.playedHand = Hand()
        # Constraint are Clubs, Diamonds ending with hcp
        self.partner_constraints = Constraints(0, 13, 0, 13, 0, 13, 0, 13, 0, 37)
        self.declarer_constraints = Constraints(0, 13, 0, 13, 0, 13, 0, 13, 0, 37)
        self.suit = bidding.get_strain_i(contract)
        self.mintricks = 13 - (int(contract[0]) + 6) + 1
        self.contract = contract
        self.tricks_taken = 0
        self.score_by_tricks_taken = [scoring.score(self.contract, is_decl_vuln, n_tricks) for n_tricks in range(14)]
        self.player_i = player_i
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
        from BGADLL import Hand
        
        self.playedHand = Hand()

    def update_trick_needed(self):
        self.mintricks += -1
        self.tricks_taken += 1
        if self.verbose:
            print("mintricks",self.mintricks)
            print("tricks_taken",self.tricks_taken)

    def set_shape_constraints(self, min_partner, max_partner, min_declarer, max_declarer, quality):
        if quality:
            margin = 1
        else:
            margin = 2

        allready_showed_partner = [0,0,0,0]
        allready_showed_partner[0] = 13 - self.partner_constraints.MaxSpades
        allready_showed_partner[1] = 13 - self.partner_constraints.MaxHearts
        allready_showed_partner[2] = 13 - self.partner_constraints.MaxDiamonds
        allready_showed_partner[3] = 13 - self.partner_constraints.MaxClubs
        allready_showed_declarer = [0,0,0,0]
        allready_showed_declarer[0] = 13 - self.declarer_constraints.MaxSpades
        allready_showed_declarer[1] = 13 - self.declarer_constraints.MaxHearts
        allready_showed_declarer[2] = 13 - self.declarer_constraints.MaxDiamonds
        allready_showed_declarer[3] = 13 - self.declarer_constraints.MaxClubs

        if self.verbose:
            print(min_partner, max_partner, min_declarer, max_declarer, quality)
            print("allready_showed_d",allready_showed_declarer)
            print("allready_showed_p",allready_showed_partner)

        for i in range(4):
            min_partner[i] = max(min_partner[i] - margin - allready_showed_partner[i], 0)
            max_partner[i] = min(max_partner[i] + margin - allready_showed_partner[i], 13)
            min_declarer[i] = max(min_declarer[i] - margin - allready_showed_declarer[i], 0)
            max_declarer[i] = min(max_declarer[i] + margin - allready_showed_declarer[i], 13)

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

        if self.verbose:
            print("set_shape_constraints")
            print("Declarer",self.declarer_constraints.ToString())
            print("Partner",self.partner_constraints.ToString())

    def set_hcp_constraints(self, min_partner, max_partner, min_declarer, max_declarer, quality):

        allready_showed_declarer = 37 - self.declarer_constraints.MaxHCP
        allready_showed_partner = 37 - self.partner_constraints.MaxHCP
        if self.verbose:
            print(min_partner, max_partner, min_declarer, max_declarer, quality)
            print("allready_showed_d",allready_showed_declarer)
            print("allready_showed_p",allready_showed_partner)
        if quality:
            margin = 2
        else:
            margin = 5
        self.declarer_constraints.MinHCP = max(min_declarer-margin-allready_showed_declarer, 0)
        self.declarer_constraints.MaxHCP = min(max_declarer+margin-allready_showed_declarer, 37)
        self.partner_constraints.MinHCP = max(min_partner-margin-allready_showed_partner, 0)
        self.partner_constraints.MaxHCP = min(max_partner+margin-allready_showed_partner, 37)

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
        self.playedHand.Add(PIMCCard(card))
        self.opposHand.Remove(PIMCCard(card))
        if playedBy == 1:
            self.dummyhand.Remove(PIMCCard(card))
        if playedBy == self.player_i:
            self.defendinghand.Remove(PIMCCard(card))
        # We will update constraints with samples after the opening lead
        if not openinglead:
            self.update_constraints(playedBy, real_card)

    async def check_threads_finished(self):
        start_time = time.time()
        while time.time() - start_time < self.wait:
            await asyncio.sleep(0.05)
            if self.pimc.Evaluating == False:
                if self.verbose:    
                    print(f"Threads are finished after {time.time() - start_time:.2f}.")
                    print(f"Playouts: {self.pimc.Playouts}")
                return
        if self.verbose:    
            print(f"Threads are still running after {self.wait} second. Examined {self.pimc.Examined} of {self.pimc.Combinations}")
        self.pimc.EndEvaluate()
        # Allow running threads to finalize
        await asyncio.sleep(0.1)
        if self.verbose:    
            print(f"Playouts: {self.pimc.Playouts}")

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
    async def nextplay(self, player_i, shown_out_suits):

        from BGADLL import Constraints, Macros, Card as PIMCCard

        self.pimc.Clear()

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

        if self.verbose:
            print("player_i", player_i)
            print(self.dummyhand.ToString(), self.defendinghand.ToString())
            print(self.opposHand.ToString(), self.playedHand.ListAsString())
            print("Voids:", shown_out_suits)
            print(Macros.Player.West if player_i == 0 else Macros.Player.East)
            print("self.player_i",self.player_i)
            print("Over dummy", self.player_i == 2)
            print("Tricks taken", self.tricks_taken)
            print("min tricks",self.mintricks)
            print("Declarer",self.declarer_constraints.ToString())
            print("Partner",self.partner_constraints.ToString())

        try:
            card = self.pimc.SetupEvaluation([self.dummyhand, self.defendinghand], self.opposHand, self.playedHand, [self.declarer_constraints,
                                  self.partner_constraints], Macros.Player.East if player_i == 2 else Macros.Player.West, self.max_playout, self.autoplay, self.player_i == 2)
        except Exception as ex:        
            print('Error:', ex)
            print("player_i", player_i)
            print(self.dummyhand.ToString(), self.defendinghand.ToString())
            print(self.opposHand.ToString(), self.playedHand.ListAsString())
            print("Voids:", shown_out_suits)
            print(Macros.Player.West if player_i == 0 else Macros.Player.East)
            print("self.player_i",self.player_i)
            print("Over dummy", self.player_i == 2)
            print("Tricks taken", self.tricks_taken)
            print("min tricks",self.mintricks)
            print("Declarer",self.declarer_constraints.ToString())
            print("Partner",self.partner_constraints.ToString())
            raise ex            
            
        
        candidate_cards = {}
        if self.autoplay and card != None:
            candidate_cards[Card.from_symbol(str(card)[::-1])] = (-1, -1, -1,"singleton - no calculation")
            return candidate_cards            

        trump = self.find_trump(self.suit)
        if self.verbose:
            print("Trump:",trump)
        self.pimc.BeginEvaluate(trump)

        await self.check_threads_finished()
        if self.verbose:
            print("Combinations:", self.pimc.Combinations)
            print("Examined:", self.pimc.Examined)
        legalMoves = self.pimc.LegalMoves
        for card in legalMoves:
            # Calculate win probability
            output = self.pimc.Output[card]
            count = float(len(output))
            # If we found no playout we need to reevaluate without constraints
            if count == 0:
                print(card)
                print(self.pimc.LegalMovesToString)
                print("Combinations:", self.pimc.Combinations)
                print("Examined:", self.pimc.Examined)
                print(self.dummyhand.ToString(), self.defendinghand.ToString())
                print(self.opposHand.ToString(), self.playedHand.ListAsString())
                print("min tricks",self.mintricks)
                print("Voids",shown_out_suits)
                print(Macros.Player.West if player_i == 0 else Macros.Player.East)
                print("self.player_i",self.player_i)
                print("Over dummy", self.player_i == 2)
                print("Declarer", self.declarer_constraints.ToString())
                print("Partner", self.partner_constraints.ToString())
                self.partner_constraints = Constraints(0, 13, 0, 13, 0, 13, 0, 13, 0, 37)
                self.declarer_constraints = Constraints(0, 13, 0, 13, 0, 13, 0, 13, 0, 37)
                print("Trying without constraints")
                return await self.nextplay(player_i, shown_out_suits)
            makable = sum(1 for t in output if t >= self.mintricks)
            probability = makable / count if count > 0 else 0
            if math.isnan(probability):
                probability = 0
            tricks = sum(t for t in output) / count if count > 0 else 0

            # Second element is the score. We need to calculate it
            score = -sum(self.score_by_tricks_taken[13 - t - self.tricks_taken] for t in output) / count if count > 0 else 0
            msg = f"{self.declarer_constraints.ToString()} - {self.partner_constraints.ToString()} - {self.pimc.Combinations} - {self.pimc.Examined} - {self.pimc.Playouts}"

            candidate_cards[Card.from_symbol(str(card)[::-1])] = (round(tricks, 2), round(score), round(probability, 2), msg)
            if self.verbose:
                print(f"{count} {Card.from_symbol(str(card)[::-1])} {tricks:.2f} {score:.0f} {probability:.2f}")

        self.pimc.EndEvaluate()
        if self.verbose:
            print(candidate_cards)
            print("Returning from PIMC")
        return candidate_cards
