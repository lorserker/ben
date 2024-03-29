import itertools
import sys
import os
import os.path
import numpy as np

from lead_binary_util import DealMeta, seats, seat_index, suit_index_lookup

from binary_righty import binary_hand, get_card_index, hot_encode_card, get_play_labels, play_data_iterator

def binary_data_decl(deal_str, outcome_str, play_str):
    x = np.zeros((1, 11, 298), np.float16)
    y = np.zeros((1, 11, 32), np.float16)
    
    hands = list(map(lambda hand_str: list(map(list, hand_str.split('.'))), deal_str[2:].split()))
    
    d_meta = DealMeta.from_str(outcome_str)

    declarer_i = seat_index[d_meta.declarer]
    dummy_i = (declarer_i + 2) % 4
    me_i = declarer_i

    dummy_bin = binary_hand(hands[dummy_i])
    me_bin = binary_hand(hands[me_i])
    
    _, on_leads, last_tricks, cards_ins, card_outs = get_play_labels(play_str, d_meta.strain, 3)
    
    dummy_played_cards = set(['>>'])
    
    for i, (on_lead, last_trick, cards_in, card_out) in enumerate(zip(on_leads, last_tricks, cards_ins, card_outs)):
        if i > 10:
            break
        label_card_ix = get_card_index(card_out)
        y[0, i, label_card_ix] = 1
        x[0, i, 292] = d_meta.level
        if d_meta.strain == 'N':
            x[0, i, 293] = 1
        else:
            x[0, i, 294 + suit_index_lookup[d_meta.strain]] = 1
        
        x[0, i, 288 + on_lead] = 1
        
        last_trick_dummy_card = last_trick[1]
        if last_trick_dummy_card not in dummy_played_cards:
            dummy_bin[get_card_index(last_trick_dummy_card)] -= 1
            dummy_played_cards.add(last_trick_dummy_card)
        
        if cards_in[1] not in dummy_played_cards:
            dummy_bin[get_card_index(cards_in[1])] -= 1
            dummy_played_cards.add(cards_in[1])
        
        x[0, i, 32:64] = dummy_bin
        x[0, i, 0:32] = me_bin
        
        x[0, i, 64:96] = hot_encode_card(last_trick[0])
        x[0, i, 96:128] = hot_encode_card(last_trick[1])
        x[0, i, 128:160] = hot_encode_card(last_trick[2])
        x[0, i, 160:192] = hot_encode_card(last_trick[3])
        
        x[0, i, 192:224] = hot_encode_card(cards_in[0])
        x[0, i, 224:256] = hot_encode_card(cards_in[1])
        x[0, i, 256:288] = hot_encode_card(cards_in[2])
        
        me_bin[label_card_ix] -= 1
        
    return x, y

if __name__ == '__main__':

    out_dir = './decl_bin_nt'

    data_it = play_data_iterator(itertools.chain(
        open('../data/play.txt'))) 

    data_it, data_it_nt, data_it_suit = itertools.tee(data_it,3)  # Create a copy of the iterator
    n1 = 0
    n2 = 0
    for i, (deal_str, outcome_str, play_str) in enumerate(data_it):
        if (i != 0) and i % 10000 == 0:
            print(i)
        d_meta = DealMeta.from_str(outcome_str)
        if d_meta.strain == "N":
            n1 += 1
        else:
            n2 += 1

    print(f"Processing {n1} deals for NT")

    X = np.zeros((n1, 11, 298), np.float16)
    Y = np.zeros((n1, 11, 32), np.float16)

    i = 0
    for j, (deal_str, outcome_str, play_str) in enumerate(data_it_nt):
        d_meta = DealMeta.from_str(outcome_str)
        if d_meta.strain == "N":
            if (i != 0) and i % 1000 == 0:
                print(i)
            x_i, y_i = binary_data_decl(deal_str, outcome_str, play_str)
            X[i, :, :] = x_i
            Y[i, :, :] = y_i
            i += 1

    np.save(os.path.join(out_dir, 'X.npy'), X)
    np.save(os.path.join(out_dir, 'Y.npy'), Y)

    out_dir = './decl_bin_suit'

    print(f"Processing {n2} deals for suit")

    X = np.zeros((n2, 11, 298), np.float16)
    Y = np.zeros((n2, 11, 32), np.float16)

    i = 0
    for j, (deal_str, outcome_str, play_str) in enumerate(data_it_suit):
        d_meta = DealMeta.from_str(outcome_str)
        if d_meta.strain != "N":
            if (i != 0) and i % 1000 == 0:
                print(i)
            x_i, y_i = binary_data_decl(deal_str, outcome_str, play_str)

            X[i, :, :] = x_i
            Y[i, :, :] = y_i
            i += 1

    np.save(os.path.join(out_dir, 'X.npy'), X)
    np.save(os.path.join(out_dir, 'Y.npy'), Y)
