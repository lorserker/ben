import itertools
import os
import os.path
import numpy as np

from binary_util import play_data_iterator, get_play_labels, DealMeta, seats, seat_index, suit_index_lookup, binary_hand, get_card_index, hot_encode_card

def binary_data_righty(deal_str, outcome_str, play_str):
    '''
    ## input
      0:32  player hand
     32:64  seen hand (dummy, or declarer if we create data for dummy)
     64:96  last trick card 0
     96:128 last trick card 1
    128:160 last trick card 2
    160:192 last trick card 3
    192:224 this trick lho card
    224:256 this trick pard card
    256:288 this trick rho card
    288:292 last trick lead player index one-hot
        292 level
    293:298 strain one hot N, S, H, D, C

    ## output
    0:32 card to play in current trick
    '''

    x = np.zeros((1, 11, 298), np.float16)
    y = np.zeros((1, 11, 32), np.float16)
    
    hands = list(map(lambda hand_str: list(map(list, hand_str.split('.'))), deal_str[2:].split()))
    d_meta = DealMeta.from_str(outcome_str)
    declarer_i = seat_index[d_meta.declarer]
    leader_i = (declarer_i + 1) % 4
    dummy_i = (declarer_i + 2) % 4
    me_i = (dummy_i + 1) % 4

    dummy_bin = binary_hand(hands[dummy_i])
    me_bin = binary_hand(hands[me_i])
    
    _, on_leads, last_tricks, cards_ins, card_outs = get_play_labels(play_str, d_meta.strain, 2)
    
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
        
        if cards_in[2] not in dummy_played_cards:
            dummy_bin[get_card_index(cards_in[2])] -= 1
            dummy_played_cards.add(cards_in[2])
        
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

def binary_data_dummy(deal_str, outcome_str, play_str):
    x = np.zeros((1, 11, 298), np.float16)
    y = np.zeros((1, 11, 32), np.float16)
    
    hands = list(map(lambda hand_str: list(map(list, hand_str.split('.'))), deal_str[2:].split()))
    d_meta = DealMeta.from_str(outcome_str)
    declarer_i = seat_index[d_meta.declarer]
    me_i = (declarer_i + 2) % 4

    declarer_bin = binary_hand(hands[declarer_i])
    me_bin = binary_hand(hands[me_i])
    
    _, on_leads, last_tricks, cards_ins, card_outs = get_play_labels(play_str, d_meta.strain, 1)
    
    decl_played_cards = set(['>>'])
    
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
        
        last_trick_decl_card = last_trick[3]
        if last_trick_decl_card not in decl_played_cards:
            declarer_bin[get_card_index(last_trick_decl_card)] -= 1
            decl_played_cards.add(last_trick_decl_card)
        
        if cards_in[1] not in decl_played_cards:
            declarer_bin[get_card_index(cards_in[1])] -= 1
            decl_played_cards.add(cards_in[1])
        
        x[0, i, 32:64] = declarer_bin
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

def binary_data_lefty(deal_str, outcome_str, play_str):
    x = np.zeros((1, 11, 298), np.float16)
    y = np.zeros((1, 11, 32), np.float16)
    
    hands = list(map(lambda hand_str: list(map(list, hand_str.split('.'))), deal_str[2:].split()))
    d_meta = DealMeta.from_str(outcome_str)
    declarer_i = seat_index[d_meta.declarer]
    dummy_i = (declarer_i + 2) % 4
    me_i = (declarer_i + 1) % 4

    dummy_bin = binary_hand(hands[dummy_i])
    me_bin = binary_hand(hands[me_i])
    
    _, on_leads, last_tricks, cards_ins, card_outs = get_play_labels(play_str, d_meta.strain, 0)
    
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
        
        if cards_in[0] not in dummy_played_cards:
            dummy_bin[get_card_index(cards_in[0])] -= 1
            dummy_played_cards.add(cards_in[0])
        
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

def handle_declarer_suit(out_dir, binary_data_decl, data_it_suit, n2):
    X = np.zeros((n2, 11, 298), np.float16)
    Y = np.zeros((n2, 11, 32), np.float16)

    i = 0
    for j, (deal_str, outcome_str, play_str) in enumerate(data_it_suit):
        d_meta = DealMeta.from_str(outcome_str)
        if d_meta.strain != "N":
            if (i != 0) and i % 10000 == 0:
                print(f"\rProgress: {i}/{n2} steps", end="")
            x_i, y_i = binary_data_decl(deal_str, outcome_str, play_str)

            X[i, :, :] = x_i
            Y[i, :, :] = y_i
            i += 1

    np.save(os.path.join(out_dir, 'X_decl_suit.npy'), X)
    np.save(os.path.join(out_dir, 'Y_decl_suit.npy'), Y)
    print()

def handle_declarer_nt(out_dir, binary_data_decl, data_it_nt, n1):
    X = np.zeros((n1, 11, 298), np.float16)
    Y = np.zeros((n1, 11, 32), np.float16)

    i = 0
    for j, (deal_str, outcome_str, play_str) in enumerate(data_it_nt):
        d_meta = DealMeta.from_str(outcome_str)
        if d_meta.strain == "N":
            if (i != 0) and i % 10000 == 0:
                print(f"\rProgress: {i}/{n1} steps", end="")
            x_i, y_i = binary_data_decl(deal_str, outcome_str, play_str)
            X[i, :, :] = x_i
            Y[i, :, :] = y_i
            i += 1

    np.save(os.path.join(out_dir, 'X_decl_nt.npy'), X)
    np.save(os.path.join(out_dir, 'Y_decl_nt.npy'), Y)
    print()

def handle_dummy_nt(out_dir, binary_data_dummy, data_it_nt, n1):
    X = np.zeros((n1, 11, 298), np.float16)
    Y = np.zeros((n1, 11, 32), np.float16)

    i = 0
    for j, (deal_str, outcome_str, play_str) in enumerate(data_it_nt):
        d_meta = DealMeta.from_str(outcome_str)
        if d_meta.strain == "N":
            if (i != 0) and i % 10000 == 0:
                print(f"\rProgress: {i}/{n1} steps", end="")
            x_i, y_i = binary_data_dummy(deal_str, outcome_str, play_str)
            X[i, :, :] = x_i
            Y[i, :, :] = y_i
            i += 1

    np.save(os.path.join(out_dir, 'X_dummy_nt.npy'), X)
    np.save(os.path.join(out_dir, 'Y_dummy_nt.npy'), Y)
    print()

def handle_dummy_suit(out_dir, binary_data_dummy, data_it_suit, n2):
    X = np.zeros((n2, 11, 298), np.float16)
    Y = np.zeros((n2, 11, 32), np.float16)

    i = 0
    for j, (deal_str, outcome_str, play_str) in enumerate(data_it_suit):
        d_meta = DealMeta.from_str(outcome_str)
        if d_meta.strain != "N":
            if (i != 0) and i % 10000 == 0:
                print(f"\rProgress: {i}/{n2} steps", end="")
            x_i, y_i = binary_data_dummy(deal_str, outcome_str, play_str)

            X[i, :, :] = x_i
            Y[i, :, :] = y_i
            i += 1

    np.save(os.path.join(out_dir, 'X_dummy_suit.npy'), X)
    np.save(os.path.join(out_dir, 'Y_dummy_suit.npy'), Y)
    print()

def handle_lefty_nt(out_dir, binary_data_lefty, data_it_nt, n1):
    X = np.zeros((n1, 11, 298), np.float16)
    Y = np.zeros((n1, 11, 32), np.float16)

    i = 0
    for j, (deal_str, outcome_str, play_str) in enumerate(data_it_nt):
        d_meta = DealMeta.from_str(outcome_str)
        if d_meta.strain == "N":
            if (i != 0) and i % 10000 == 0:
                print(f"\rProgress: {i}/{n1} steps", end="")
            x_i, y_i = binary_data_lefty(deal_str, outcome_str, play_str)
            X[i, :, :] = x_i
            Y[i, :, :] = y_i
            i += 1

    np.save(os.path.join(out_dir, 'X_lefty_nt.npy'), X)
    np.save(os.path.join(out_dir, 'Y_lefty_nt.npy'), Y)
    print()

def handle_lefty_suit(out_dir, binary_data_lefty, data_it_suit, n2):
    X = np.zeros((n2, 11, 298), np.float16)
    Y = np.zeros((n2, 11, 32), np.float16)

    i = 0
    for j, (deal_str, outcome_str, play_str) in enumerate(data_it_suit):
        d_meta = DealMeta.from_str(outcome_str)
        if d_meta.strain != "N":
            if (i != 0) and i % 10000 == 0:
                print(f"\rProgress: {i}/{n2} steps", end="")
            x_i, y_i = binary_data_lefty(deal_str, outcome_str, play_str)

            X[i, :, :] = x_i
            Y[i, :, :] = y_i
            i += 1

    np.save(os.path.join(out_dir, 'X_lefty_suit.npy'), X)
    np.save(os.path.join(out_dir, 'Y_lefty_suit.npy'), Y)
    print()

def handle_righty_nt(out_dir, binary_data_righty, data_it_nt, n1):
    X = np.zeros((n1, 11, 298), np.float16)
    Y = np.zeros((n1, 11, 32), np.float16)

    i = 0
    for j, (deal_str, outcome_str, play_str) in enumerate(data_it_nt):
        d_meta = DealMeta.from_str(outcome_str)
        if d_meta.strain == "N":
            if (i != 0) and i % 10000 == 0:
                print(f"\rProgress: {i}/{n1} steps", end="")
            x_i, y_i = binary_data_righty(deal_str, outcome_str, play_str)
            X[i, :, :] = x_i
            Y[i, :, :] = y_i
            i += 1

    np.save(os.path.join(out_dir, 'X_righty_nt.npy'), X)
    np.save(os.path.join(out_dir, 'Y_righty_nt.npy'), Y)
    print()    

def handle_righty_suit(out_dir, binary_data_righty, data_it_suit, n2):
    X = np.zeros((n2, 11, 298), np.float16)
    Y = np.zeros((n2, 11, 32), np.float16)

    i = 0
    for j, (deal_str, outcome_str, play_str) in enumerate(data_it_suit):
        d_meta = DealMeta.from_str(outcome_str)
        if d_meta.strain != "N":
            if (i != 0) and i % 10000 == 0:
                print(f"\rProgress: {i}/{n2} steps", end="")
            x_i, y_i = binary_data_righty(deal_str, outcome_str, play_str)

            X[i, :, :] = x_i
            Y[i, :, :] = y_i
            i += 1

    np.save(os.path.join(out_dir, 'X_righty_suit.npy'), X)
    np.save(os.path.join(out_dir, 'Y_righty_suit.npy'), Y)
    print()

if __name__ == '__main__':

    data_it = play_data_iterator(itertools.chain(
        open('../../data/WBC/play.txt'))) 
    out_dir = './binary'

    # Using data from Jack
    # data_it = play_data_iterator(itertools.chain(
    #     open('../../data/jack/BW5C_N.txt'),
    #     open('../../data/jack/BW5C_S.txt'),
    #     open('../../data/jack/JOS_N.txt'),
    #     open('../../data/jack/JOS_S.txt')
    # ))
    # out_dir = './binary_jack'

    data_it_play = itertools.tee(data_it,9)  # Create a copy of the iterator
    print("Counting deals")
    n1 = 0
    n2 = 0
    for i, (deal_str, outcome_str, play_str) in enumerate(data_it_play[0]):
        if (i != 0) and i % 100000 == 0:
            print(f"\rProgress: {i} deals", end="")
        d_meta = DealMeta.from_str(outcome_str)
        if d_meta.strain == "N":
            n1 += 1
        else:
            n2 += 1
    print()
    print("{0} total deals".format(n1+n2))

    print(f"Processing {n1} deals for NT")

    print("Declarer")
    handle_declarer_nt(out_dir, binary_data_decl, data_it_play[1], n1)
    print("Dummy")
    handle_dummy_nt(out_dir, binary_data_dummy, data_it_play[2], n1)
    print("Left hand opponent")
    handle_lefty_nt(out_dir, binary_data_lefty, data_it_play[3], n1)
    print("Right hand opponent")
    handle_righty_nt(out_dir, binary_data_righty, data_it_play[4], n1)

    print(f"Processing {n2} deals for suit")

    print("Declarer")
    handle_declarer_suit(out_dir, binary_data_decl, data_it_play[5], n2)
    print("Dummy")
    handle_dummy_suit(out_dir, binary_data_dummy, data_it_play[6], n2)
    print("Left hand opponent")
    handle_lefty_suit(out_dir, binary_data_lefty, data_it_play[7], n2)
    print("Right hand opponent")
    handle_righty_suit(out_dir, binary_data_righty, data_it_play[8], n2)
