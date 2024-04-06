import numpy as np


def find_nth_occurrence(arr, target, nth):
    counter = 0
    idx = 0

    while idx < len(arr) and counter <= nth:
        if arr[idx] == target:
            counter += 1

        if counter == nth + 1:
            return idx - 1

        idx += 1

    raise ValueError(f"Could not find the {nth}-th occurrence of '{target}' within the array.")


def select_right_card(hand52, opening_lead, rng, contract, models):
    if contract[1] != "N":
        opening_suit = opening_lead // 8
        if opening_suit == "SHDC".index(contract[1]):
            # In trump always lead lowest from pips
            card_index = 13 - 1 - np.nonzero(np.flip(hand52.reshape((4, 13))[opening_suit]))[0][0]
            return card_index + 13 * opening_suit
    else:
        opening_suit = opening_lead // 8
        suit_length = len(hand52.reshape((4, 13))[opening_suit])
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
