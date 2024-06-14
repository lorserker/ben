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


def select_right_card_for_play(candidate_cards, rng, contract, models, hand_str, dummy_str, player_i, tricks52, current_trick, play_status, who, verbose):
    #print("select_right_card_for_play")
    if len(candidate_cards) == 1:
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
            plays = response_dict["SuitCAnalysis"]["OptimumPlays"]
            if len(plays) > 0:
                #print("plays", len(plays))
                card = plays[0]["GameTree"]["T"]
                #print("card", card)
            else:
                print("SuitC found no plays")
                return candidate_cards[0].card, who
            if card == "N/A":
                return candidate_cards[0].card, who
            else:
                card = card[-1]
            suit_str = "SHDC"[interesting_suit]
            print("SuitC playing:","SHDC"[interesting_suit], card)
            for candidate_card in candidate_cards:
                if candidate_card.card.symbol() == f"{suit_str}{card}":

                    return candidate_card.card, "SuitC"
                    
        return candidate_cards[0].card, who
    
    
    #print(candidate_cards[0].card)
    #print(candidate_cards[0].card.code())
    if original_count == current_count:
        if candidate_cards[0].card.code() % 13 >= 7 and len(current_trick) == 0:
            hand52 = binary.parse_hand_f(52)(hand_str)
            # if we havent't lead from that suit then lead as opening lead, but only for pips
            card = select_right_card(hand52, deck52.card52to32(candidate_cards[0].card.code()), rng, contract, models)    
            return Card.from_code(card), who
    
    return candidate_cards[0].card, who


def select_right_card(hand52, opening_lead, rng, contract, models):
    #print("select_right_card")
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
            # In trump always lead lowest from pips
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

        
