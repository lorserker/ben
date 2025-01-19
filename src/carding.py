import sys
from urllib.request import urlopen
import json
import numpy as np

from objects import Card
import deck52
import binary
from util import save_for_suitc
from colorama import Fore, Back, Style, init

init()

RANKS = 'AKQJT98765432'

def find_nth_occurrence(arr, target, nth):
    counter = 0
    idx = 0

    while idx < len(arr) and counter <= nth:
        if arr[idx] == target:
            counter += 1

        if counter == nth:
            return idx

        idx += 1

    raise ValueError(f"Could not find the {nth}-th occurrence of '{target}' within the array.")
def find_last_occurrence(arr, target):
    counter = len(arr) -1

    while  counter >= 0:
        if arr[counter] != target:
            counter -= 1
        else:
            return counter

    raise ValueError(f"Could not find the last occurrence of '{target}' within the array.")

def count_entries(hand_str, interesting_suit, played_cards):
    #print("Played cards", played_cards)
    suits = hand_str.split(".")
    entry_values = {'A': 1, 'K': 1, 'Q': 0.5}

    total_entries = 0
    for i, suit in enumerate(suits):
        if i != interesting_suit:
            for card in suit:
                if card in played_cards[i]:
                    continue
                total_entries += entry_values.get(card, 0)

    #print("Total entries", total_entries, hand_str, interesting_suit)
    return round(total_entries)    

def select_right_card_for_play(candidate_cards, rng, contract, models, hand_str, dummy_str, player_i, tricks52, current_trick, missing_cards, play_status, who, verbose):
    if verbose:
        print(f"select_right_card_for_play for player {player_i} {hand_str} Play status: {play_status}")
    if len(candidate_cards) == 1:
        return candidate_cards[0].card, who
    
    if play_status == "Discard":
        # As we dont give count, just discard smallest if dummy
        # consider other rules in trump
        if player_i == 1:
            #print("play from dummy")
            if candidate_cards[0].card.code() % 13 >= 7:
                # we need to remove already played cards
                hand52 = binary.parse_hand_f(52)(hand_str)
                for trick in tricks52:
                    for card in trick:
                        if hand52[0][card] == 1:
                            hand52[0][card] = 0

                card_index = find_last_occurrence(hand52.reshape((4, 13))[candidate_cards[0].card.suit],1)
                card_index = card_index + 13 * candidate_cards[0].card.suit
                return Card.from_code(card_index), "Discarding"
            
        if player_i == 3:
            # Currently no specific rule for declarer
            pass

        if player_i == 0 or player_i == 2:
            # Currently no specific rule for defenders
            pass

    # Perhaps this should be removed, second card can be from another suit, so not sure difference make sense
    if not models.force_suitc:
            # If the first card is better then don't evaluate
        if hasattr(candidate_cards[0], 'p_make_contract'):
            if candidate_cards[0].p_make_contract > candidate_cards[1].p_make_contract + 0.1:
                return candidate_cards[0].card, who
        if hasattr(candidate_cards[0], 'expected_tricks_dd'):
            if candidate_cards[0].expected_tricks_dd > candidate_cards[1].expected_tricks_dd + 1:
                return candidate_cards[0].card, who
        
        if hasattr(candidate_cards[0], 'expected_score_mp') and candidate_cards[0].expected_score_mp is not None:
            if candidate_cards[0].expected_score_mp > candidate_cards[1].expected_score_mp + 5:
                return candidate_cards[0].card, who
        if hasattr(candidate_cards[0], 'expected_score_imp') and candidate_cards[0].expected_score_imp is not None:
            if candidate_cards[0].expected_score_imp > candidate_cards[1].expected_score_imp + 1:
                return candidate_cards[0].card, who
    
    interesting_suit = candidate_cards[0].card.suit
    #print("interesting_suit", interesting_suit)
    if verbose:
        print(f'Checking SuitC: {models.use_suitc} {contract}  {models.suitc_sidesuit_check or "SHDC"[interesting_suit] == contract[1]}')

    if player_i == 3 and not models.use_suitc:
        # For declarer pick a random card, when touching honors and NN is equal (Will not happen in practice)
        #print("First card for declarer", candidate_cards[0].card)
        #print(hand_str)
        #print(dummy_str)
        #print("play_status", play_status)
        return candidate_cards[0].card, who       

    suits = hand_str.split('.')
    original_count = len(suits[interesting_suit])
    current_count  = original_count
    #print("original_count", original_count, suits[interesting_suit])
    discards = ""
    for trick in tricks52:
        for card in trick:
            if card // 13 == interesting_suit:
                c = RANKS[card % 13]
                if c in suits[interesting_suit]:
                    current_count -= 1
                else:
                    discards += c

    if models.use_suitc:
        if contract[1] == "N" or models.suitc_sidesuit_check or "SHDC"[interesting_suit] == contract[1]:
            if (player_i  == 1 or player_i == 3) and play_status == "Lead":
                # We only use SuitC the first time the suit is played 
                # but allow 3 discards / rufs in the suit
                if current_count + 2 >= original_count and len(discards) < 3 and models.use_suitc:
                    if verbose:
                        print("SuitC activated")
                        print("discards", discards)
                        print("current_count", current_count)
                        print("original_count", original_count)
                        print("tricks52",tricks52)
                    # For dummy just take lowest card. Could be stressing opponents by taking highest of touching cards.
                    #print("First card for dummy", candidate_cards[0].card)
                    suits_north = dummy_str.split('.')[interesting_suit]
                    suits_south = suits[interesting_suit]
                    #suits_west = ""
                    #suits_east = ""
                    suits_westeast = ""
                    if verbose: 
                        print(f"SuitC: {suits_north} {suits_south} {suits_westeast}")
                        print("discards", discards)
                    for c in RANKS:
                        if c in suits_north:
                            continue
                        if c in suits_south:
                            continue
                        if c in discards:
                            continue
                        suits_westeast  += c

                    if len(suits_westeast) < 2:
                        if verbose:
                            print("Opponents got at most one card in the suit, no calculations")
                        return candidate_cards[0].card, who 
                    if len(suits_westeast) > 6 and "SHDC"[interesting_suit] != contract[1]:
                        if verbose:
                            print("Opponents got more cards in the suit than us, no calculations")
                        return candidate_cards[0].card, who 
                    from suitc.SuitC import SuitCLib
                    suitc = SuitCLib(verbose)
                    # We just use a simple version of entries
                    
                    played_cards = [[], [], [], []]
                    for trick in tricks52:
                        for card in trick:
                            played_cards[card // 13].append(RANKS[card % 13])   

                    entries = count_entries(hand_str, interesting_suit, played_cards)

                    card = suitc.calculate(f"{suits_north if suits_north != '' else '.'} {suits_south if suits_south != '' else '.'} {suits_westeast}", trump = "SHDC"[interesting_suit] == contract[1], entries = entries )
                    suitc_card = None
                    try:
                        response_dict = json.loads(card)
                        optimum_plays = response_dict["SuitCAnalysis"]["OptimumPlays"]
                        # We just take the play for MAX as we really don't know how many tricks are needed
                        for play in optimum_plays:
                            # If we can take all tricks we drop SuitC
                            #print(play['Plays'][0]['Tricks'])
                            #print("Tricks", play['Plays'][0]['Tricks'], " Max: ",max(len(suits_north),len(suits_south)))
                            if play['Plays'][0]['Tricks'] == max(len(suits_north),len(suits_south)):
                                if play['Plays'][0]['Percentage'] == 100:
                                    if verbose:
                                        print(f"SuitC dropped as we can take all tricks {current_count} {original_count} ")
                                    return candidate_cards[0].card, who
                            # We can have more than one play for MAX
                            # So currently we are then selecting lowest card. Should that be different
                            if "MAX" in play["OptimumPlayFor"]:
                                if len(play["GameTree"]) > 0:
                                    suitc_card = play["GameTree"]["T"]
                                else:
                                    print("SuitC found no gametree")
                                    return candidate_cards[0].card, who
                                #print("card", card)
                    except:
                        sys.stderr.write(f"{Fore.RED}SuitC failed: {card} Input:{suits_north if suits_north != '' else '.'} {suits_south if suits_south != '' else '.'} {suits_westeast}{Fore.RESET}\n")
                        return candidate_cards[0].card, who
                    if suitc_card is None:
                        if verbose:
                            print("SuitC found no plays")
                        return candidate_cards[0].card, who
                    else:
                        suitc_card = suitc_card[-1]
                    suit_str = "SHDC"[interesting_suit]
                    if verbose:
                        print(f"SuitC found: {suit_str}{suitc_card}")
                    # Only play the card from SuitC if it was a candidate
                    for candidate_card in candidate_cards:
                        if candidate_card.card.symbol() == f"{suit_str}{suitc_card}":
                            # Only play SuitC if not losing to much DD
                            if models.use_real_imp_or_mp:
                                if models.matchpoint:
                                    if candidate_card.expected_score_mp >= candidate_cards[0].expected_score_mp - 4:
                                        return candidate_card.card, "SuitC-MP"
                                else:
                                    if candidate_card.expected_score_imp >= candidate_cards[0].expected_score_imp - 0.4:
                                        return candidate_card.card, "SuitC-Imp"
                            else:
                                if models.double_dummy:
                                    if candidate_card.p_make_contract >= candidate_cards[0].p_make_contract - 0.1:
                                        if candidate_card.expected_tricks_dd and candidate_card.expected_tricks_dd >= candidate_cards[0].expected_tricks_dd - 0.2:
                                            return candidate_card.card, "SuitC-DD"
                                else:
                                    if candidate_card.p_make_contract >= candidate_cards[0].p_make_contract - 0.1:
                                        if candidate_card.expected_tricks_sd and candidate_card.expected_tricks_sd >= candidate_cards[0].expected_tricks_sd - 0.2:
                                            return candidate_card.card, "SuitC-SD"
                            if verbose:
                                print("SuitC candidate card worse than best DD cards")
                                print("SuitC card", candidate_card)
                                print("DD card", candidate_cards[0])
                            save_for_suitc(suits_north, suits_south, candidate_card, candidate_cards[0], optimum_plays, hand_str, dummy_str)
                            if models.force_suitc:
                                return candidate_card.card, "SuitC-Forced"
                return candidate_cards[0].card, who
    
    if original_count == current_count:
        # If we are on lead, and playing the suit the first time, follow our lead agreements.
        # We should probably look at first card in each tricks52
        #print("First card for dummy", candidate_cards[0].card)
        if candidate_cards[0].card.code() % 13 >= 7 and play_status == "Lead":
            # if we havent't lead from that suit then lead as opening lead, but only for pips
            # If dummy and leading a pip, just lead lowest
            hand52 = binary.parse_hand_f(52)(hand_str)
            if player_i == 1:
                if verbose:
                    print("Lead lowest pip from dummy")
                card_index = find_last_occurrence(hand52.reshape((4, 13))[candidate_cards[0].card.suit],1)
                card_index = card_index + 13 * candidate_cards[0].card.suit
                return Card.from_code(card_index), who

            card = select_right_card(hand52, deck52.card52to32(candidate_cards[0].card.code()), rng, contract, models, verbose)    
            if verbose:
                print("card selected", card)
            return Card.from_code(card), who
    # If we are dummy and playing a pip, just play lowest
    if player_i == 1 and play_status == "Follow":
        return select_lowest_card_dynamic(candidate_cards), who

    if verbose:
        print("No lead rules actual, playing top of candidate list", candidate_cards[0].card)
    return candidate_cards[0].card, who

def select_lowest_card_dynamic(candidates):
    # Get the top card and its suit
    top_card = candidates[0]
    top_suit = top_card.card.suit  # Adjusted for Card class

    # Get all attributes from the top card except `None`, `card`, and `msg`
    comparison_keys = {
        attr: getattr(top_card, attr)
        for attr in dir(top_card)
        if not callable(getattr(top_card, attr)) 
           and not attr.startswith("_") 
           and getattr(top_card, attr) is not None 
           and attr not in {"card", "msg"}  # Exclude `card` and `msg`
    }

    # Filter candidates matching criteria
    def matches_criteria(candidate):
        # Same suit as the top card
        if candidate.card.suit != top_suit:
            return False
        # Compare all attributes present in the top card
        for key, value in comparison_keys.items():
            if getattr(candidate, key, None) != value:
                return False
        return True

    matching_cards = [
        candidate.card for candidate in candidates if matches_criteria(candidate)
    ]
    
    # Sort by card rank using Card.RANKS
    matching_cards.sort(key=lambda card: card.rank, reverse=True)
    
    # Return the lowest card
    return matching_cards[0] if matching_cards else None

def select_right_card(hand52, opening_lead, rng, contract, models, verbose):
    if verbose:
        print("Select the right card")
    opening_suit = opening_lead // 8
    suit_length = np.sum(hand52.reshape((4, 13))[opening_suit])
    if contract[1] == "N":
        if models.lead_from_pips_nt == "attitude" and suit_length > 1:
            # Need to check for honor in the suit
            hcp_in_suit = binary.get_hcp_suit(hand52.reshape((4, 13))[opening_suit])
            if suit_length < 4 or hcp_in_suit < 2:
                card_index = find_nth_occurrence(hand52.reshape((4, 13))[opening_suit], 1, 1)
            else:
                card_index = find_nth_occurrence(hand52.reshape((4, 13))[opening_suit], 1, suit_length)
            return card_index + 13 * opening_suit
        
        if models.lead_from_pips_nt == "24" and suit_length > 1:
            # From touching honors lead the highest
            # From bad suit lead 2nd highest
            hcp_in_suit = binary.get_hcp_suit(hand52.reshape((4, 13))[opening_suit])
            if suit_length < 4 or hcp_in_suit < 1:
                card_index = find_nth_occurrence(hand52.reshape((4, 13))[opening_suit], 1, 2)
            else:
                card_index = find_nth_occurrence(hand52.reshape((4, 13))[opening_suit], 1, 4)
            return card_index + 13 * opening_suit
    else:
        if opening_suit == "SHDC".index(contract[1]):
            # when leading a trump always lead lowest from pips
            card_index = 13 - 1 - np.nonzero(np.flip(hand52.reshape((4, 13))[opening_suit]))[0][0]
            return card_index + 13 * opening_suit

        if models.lead_from_pips_suit == "135":
            if suit_length < 3:
                card_index = find_nth_occurrence(hand52.reshape((4, 13))[opening_suit], 1, 1)
            else:
                if suit_length < 5:
                    card_index = find_nth_occurrence(hand52.reshape((4, 13))[opening_suit], 1, 3)
                else:
                    card_index = find_nth_occurrence(hand52.reshape((4, 13))[opening_suit], 1, 5)
            return card_index + 13 * opening_suit

    #Perhaps this is the right place to add generic rules like 2nd hand low and 3rd hand high

    # Fallback to the original lead of a random pip
    # it's a pip ~> choose a random one
    if opening_lead % 8 == 7:
        pips_mask = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        lefty_led_pips = hand52.reshape((4, 13))[opening_lead // 8] * pips_mask
        opening_lead52 = (opening_lead // 8) * 13 + rng.choice(np.nonzero(lefty_led_pips)[0])

    else:
        opening_lead52 = deck52.card32to52(opening_lead)

    return opening_lead52

        
