import numpy as np

from objects import Card
import deck52
import binary


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


def select_right_card_for_play(candidate_cards, rng, contract, models, hand_str, dummy_str, player_i, played_cards, current_trick, play_status):
    print("select_right_card_for_play")
    if len(candidate_cards) == 1:
        return candidate_cards[0].card
    
    if player_i  == 1:
        # For dummy just take lowest card. Could be stressing opponents by taking highest of touching cards.
        print("First card for dummy", candidate_cards[0].card)
        print(hand_str)
        print(dummy_str)
        print("play_status", play_status)
        return candidate_cards[0].card
    if player_i == 3:
        # For declarer pick a random card, when touching honors and NN is equal (Will not happen in practice)
        print("First card for declarer", candidate_cards[0].card)
        print(hand_str)
        print(dummy_str)
        print("play_status", play_status)
        return candidate_cards[0].card       
    suit = candidate_cards[0].card.code() // 13
    hand52 = binary.parse_hand_f(52)(hand_str)
    original_count = np.sum(hand52.reshape((4, 13))[suit])
    current_count= np.sum(played_cards.reshape((4, 8))[suit])
    # print("original_count", original_count)
    # print("current_count", current_count)
    # print("public_cards",played_cards)
    # print(candidate_cards[0].card)
    # print(candidate_cards[0].card.code())
    if original_count == current_count:
        if candidate_cards[0].card.code() % 13 >= 7 and len(current_trick) == 0:
            # if we havent't lead from that suit then lead as opening lead, but only for pips
            card = select_right_card(hand52, deck52.card52to32(candidate_cards[0].card.code()), rng, contract, models)    
            return Card.from_code(card)
    
    return candidate_cards[0].card


def select_right_card(hand52, opening_lead, rng, contract, models):
    print("select_right_card")
    if contract[1] != "N":
        opening_suit = opening_lead // 8
        if opening_suit == "SHDC".index(contract[1]):
            # In trump always lead lowest from pips
            card_index = 13 - 1 - np.nonzero(np.flip(hand52.reshape((4, 13))[opening_suit]))[0][0]
            return card_index + 13 * opening_suit
    else:
        opening_suit = opening_lead // 8
        suit_length = np.sum(hand52.reshape((4, 13))[opening_suit])
        if models.lead_from_pips_nt == "attitude" and suit_length > 1:
            # Need to check for honor in the suit
            if suit_length < 4:
                card_index = find_nth_occurrence(hand52.reshape((4, 13))[opening_suit], 1, 1)
            else:
                card_index = find_nth_occurrence(hand52.reshape((4, 13))[opening_suit], 1, suit_length)
            return card_index + 13 * opening_suit
        if models.lead_from_pips_nt == "24" and suit_length > 1:
            if suit_length < 4:
                card_index = find_nth_occurrence(hand52.reshape((4, 13))[opening_suit], 1, 2)
            else:
                card_index = find_nth_occurrence(hand52.reshape((4, 13))[opening_suit], 1, 4)
            return card_index + 13 * opening_suit
        if models.lead_from_pips_nt == "135":
            if suit_length < 3:
                card_index = find_nth_occurrence(hand52.reshape((4, 13))[opening_suit], 1, 1)
            else:
                if suit_length < 5:
                    card_index = find_nth_occurrence(hand52.reshape((4, 13))[opening_suit], 1, 3)
                else:
                    card_index = find_nth_occurrence(hand52.reshape((4, 13))[opening_suit], 1, 5)
            return card_index + 13 * opening_suit

    # it's a pip ~> choose a random one
    pips_mask = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    lefty_led_pips = hand52.reshape((4, 13))[opening_lead // 8] * pips_mask
    opening_lead52 = (opening_lead // 8) * 13 + rng.choice(np.nonzero(lefty_led_pips)[0])
    return opening_lead52
