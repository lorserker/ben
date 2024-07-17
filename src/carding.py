from urllib.request import urlopen
import json
import numpy as np

from objects import Card
import deck52
import binary

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


def select_right_card_for_play(candidate_cards, rng, contract, models, hand_str, dummy_str, player_i, tricks52, current_trick, play_status, who, verbose):
    if verbose:
        print(f"select_right_card_for_play fo for player {player_i} {hand_str}")
    if len(candidate_cards) == 1:
        return candidate_cards[0].card, who
    
    if verbose:
        print(candidate_cards[0])

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
                return Card.from_code(card_index), who
            
        if player_i == 3:
            # Currently no specific rule for declarer
            pass

    # If the first card is better then don't evaluate
    if candidate_cards[0].p_make_contract > candidate_cards[1].p_make_contract + 0.05:
        return candidate_cards[0].card, who
    if candidate_cards[0].expected_tricks_dd > candidate_cards[1].expected_tricks_dd + 0.5:
        return candidate_cards[0].card, who
    
    if player_i == 3 and not models.use_suitc:
        # For declarer pick a random card, when touching honors and NN is equal (Will not happen in practice)
        #print("First card for declarer", candidate_cards[0].card)
        #print(hand_str)
        #print(dummy_str)
        #print("play_status", play_status)
        return candidate_cards[0].card, who       

    interesting_suit = candidate_cards[0].card.suit
    #print("interesting_suit", interesting_suit)
    suits = hand_str.split('.')
    original_count = len(suits[interesting_suit])
    current_count  = original_count
    #print("original_count", original_count, suits[interesting_suit])
    discards = ""
    for trick in tricks52:
        for card in trick:
            if card // 13 == interesting_suit:
                #print(suits[interesting_suit])
                c = RANKS[card % 13]
                if c in suits[interesting_suit]:
                    #print("found", c, card)
                    current_count -= 1
                    break
                else:
                    discards += c
    if contract[1] == "N" or models.suitc_sidesuit_check or interesting_suit == contract[1]:
        if (player_i  == 1 or player_i == 3) and play_status == "Lead":
            # We only use SuitC the first time the suit is played - should probably look at first card in each tricks52
            if original_count == current_count and models.use_suitc:
                print("SuitC activated")
                if verbose:
                    print("discards", discards)
                    print("current_count", current_count)
                    print("tricks52",tricks52)
                # For dummy just take lowest card. Could be stressing opponents by taking highest of touching cards.
                #print("First card for dummy", candidate_cards[0].card)
                suits_north = dummy_str.split('.')[interesting_suit]
                suits_south = suits[interesting_suit]
                #suits_west = ""
                #suits_east = ""
                suits_westeast = ""
                for c in RANKS:
                    if c in suits_north:
                        continue
                    if c in suits_south:
                        continue
                    if c in discards:
                        continue
                    suits_westeast  += c

                from suitc.SuitC import SuitCLib
                suitc = SuitCLib(False)

                card = suitc.calculate(f"{suits_north if suits_north != '' else '.'} {suits_south} {suits_westeast}")
                
                response_dict = json.loads(card)
                optimum_plays = response_dict["SuitCAnalysis"]["OptimumPlays"]
                card == "N/A"
                # We just take the play for MAX as we really don't know how many tricks are needed
                for play in optimum_plays:
                    if "MAX" in play["OptimumPlayFor"]:
                        if len(play["GameTree"]) > 0:
                            card = play["GameTree"]["T"]
                        else:
                            print("SuitC found no gametree")
                            return candidate_cards[0].card, who
                        #print("card", card)
                if card == "N/A":
                    print("SuitC found no plays")
                    return candidate_cards[0].card, who
                else:
                    card = card[-1]
                suit_str = "SHDC"[interesting_suit]
                print(f"SuitC found: {suit_str}{card}")
                # Only play the card from SuitC if it was a candidate
                for candidate_card in candidate_cards:
                    if candidate_card.card.symbol() == f"{suit_str}{card}":
                        # Only play SuitC if not losing to much DD
                        if candidate_card.p_make_contract > candidate_cards[0].p_make_contract - 0.05:
                            if candidate_card.expected_tricks_dd > candidate_cards[0].expected_tricks_dd - 0.5:
                                return candidate_card.card, "SuitC"
                        
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
                print("Lead lowest pip")
                card_index = find_last_occurrence(hand52.reshape((4, 13))[candidate_cards[0].card.suit],1)
                card_index = card_index + 13 * candidate_cards[0].card.suit
                return Card.from_code(card_index), who

            card = select_right_card(hand52, deck52.card52to32(candidate_cards[0].card.code()), rng, contract, models, verbose)    
            if verbose:
                print("card selected", card)
            return Card.from_code(card), who
    if verbose:
        print("No lead rules actual, playing top of candidate list", candidate_cards[0].card)
    return candidate_cards[0].card, who


def select_right_card(hand52, opening_lead, rng, contract, models, verbose):
    if verbose:
        print("Select the right card")
    opening_suit = opening_lead // 8
    suit_length = np.sum(hand52.reshape((4, 13))[opening_suit])
    if contract[1] == "N":
        if models.lead_from_pips_nt == "attitude" and suit_length > 1:
            # Need to check for honor in the suit
            if suit_length < 4:
                card_index = find_nth_occurrence(hand52.reshape((4, 13))[opening_suit], 1, 1)
            else:
                card_index = find_nth_occurrence(hand52.reshape((4, 13))[opening_suit], 1, suit_length)
            return card_index + 13 * opening_suit
        
        if models.lead_from_pips_nt == "24" and suit_length > 1:
            # From touching honors lead the highest
            if suit_length < 4:
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

    # Fallback to the original lead of a random pip
    # it's a pip ~> choose a random one
    if opening_lead % 8 == 7:
        pips_mask = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        lefty_led_pips = hand52.reshape((4, 13))[opening_lead // 8] * pips_mask
        opening_lead52 = (opening_lead // 8) * 13 + rng.choice(np.nonzero(lefty_led_pips)[0])

    else:
        opening_lead52 = deck52.card32to52(opening_lead)

    return opening_lead52

        
