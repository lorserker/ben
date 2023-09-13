import numpy as np
import itertools
import os
import os.path

from binary_righty import binary_hand, get_card_index
from lead_binary_util import DealMeta
from binary_lead import jack_data_iterator


def create_binary(data_it, n, out_dir):
    
    X = np.zeros((n, 32 + 5 + 4*32), dtype=np.float16)
    y = np.zeros((n, 14), dtype=np.float16)

    for i, (deal_str, meta_str, auction_str, play_str) in enumerate(data_it):
        if i % 1000 == 0:
            print(i)

        deal_meta = DealMeta.from_str(meta_str)

        declarer_ix = 'WNES'.index(deal_meta.declarer)
        hands_str = list(map(lambda hand_str: list(map(list, hand_str.split('.'))), deal_str[2:].split()))

        lead_card = play_str[:2]
        lead_card_ix = get_card_index(lead_card)
        strain_ix = 'NSHDC'.index(deal_meta.strain)

        y[i, deal_meta.tricks_made] = 1

        X[i, lead_card_ix] = 1
        X[i, 32 + strain_ix] = 1
        # lefty
        X[i, (32 + 5 + 0*32):(32 + 5 + 1*32)] = binary_hand(hands_str[(declarer_ix + 1) % 4])
        # dummy
        X[i, (32 + 5 + 1*32):(32 + 5 + 2*32)] = binary_hand(hands_str[(declarer_ix + 2) % 4])
        # righty
        X[i, (32 + 5 + 2*32):(32 + 5 + 3*32)] = binary_hand(hands_str[(declarer_ix + 3) % 4])
        # declarer
        X[i, (32 + 5 + 3*32):] = binary_hand(hands_str[declarer_ix])

    np.save(os.path.join(out_dir, 'X.npy'), X)
    np.save(os.path.join(out_dir, 'y.npy'), y)


if __name__ == '__main__':

    out_dir = './lr3_bin'

    data_it = jack_data_iterator(itertools.chain(
        open('playing_data/jack/BW5C_N.txt'), 
        open('playing_data/jack/BW5C_S.txt'),
        open('playing_data/jack/JOS_N.txt'),
        open('playing_data/jack/JOS_S.txt')))

    n = 633873

    create_binary(data_it, n, out_dir)
