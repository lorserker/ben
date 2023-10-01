import time

import numpy as np

import binary
import conf

from bidding import bidding
from util import get_all_hidden_cards
from configparser import ConfigParser
from util import hand_to_str


def get_small_out_i(small_out):
    x = small_out.copy()
    dec = np.minimum(1, x)

    result = []
    while np.max(x) > 0:
        result.extend(np.nonzero(x)[0])
        x = x - dec
        dec = np.minimum(1, x)

    return result


def distr_vec(x):
    xpos = np.maximum(x, 0) + 0.1
    pvals = xpos / np.sum(xpos, axis=1, keepdims=True)

    p_cumul = np.cumsum(pvals, axis=1)

    indexes = np.zeros(pvals.shape[0], dtype=np.int32)
    rnd = np.random.rand(pvals.shape[0])

    for k in range(p_cumul.shape[1]):
        indexes = indexes + (rnd > p_cumul[:, k])

    return indexes


def distr2_vec(x1, x2):
    x1pos = np.maximum(x1, 0) + 0.1
    x2pos = np.maximum(x2, 0) + 0.1

    pvals1 = x1pos / np.sum(x1pos, axis=1, keepdims=True)
    pvals2 = x2pos / np.sum(x2pos, axis=1, keepdims=True)

    pvals = pvals1 * pvals2
    pvals = pvals / np.sum(pvals, axis=1, keepdims=True)

    return distr_vec(pvals)


def player_to_nesw_i(player_i, contract):
    decl_i = bidding.get_decl_i(contract)
    return (decl_i + player_i + 1) % 4


class Sample:

    def __init__(self, lead_accept_threshold, bidding_threshold_sampling, play_accept_threshold, bid_accept_play_threshold, sample_hands_auction, sample_boards_for_auction, sample_boards_for_auction_lead, sample_hands_opening_lead, sample_hands_play, verbose):
        self.lead_accept_threshold = lead_accept_threshold
        self.bidding_threshold_sampling = bidding_threshold_sampling
        self.play_accept_threshold = play_accept_threshold
        self.bid_accept_play_threshold = bid_accept_play_threshold
        self._sample_hands_auction = sample_hands_auction
        self.sample_boards_for_auction = sample_boards_for_auction
        self.sample_boards_for_auction_lead = sample_boards_for_auction_lead
        self.sample_hands_opening_lead = sample_hands_opening_lead
        self.sample_hands_play = sample_hands_play
        self.verbose = verbose

    @classmethod
    def from_conf(cls, conf: ConfigParser, verbose= False) -> "Sample":
        lead_accept_threshold = float(conf['sampling']['lead_accept_threshold'])
        bidding_threshold_sampling = float(conf['sampling']['bidding_threshold_sampling'])
        play_accept_threshold = float(conf['sampling']['play_accept_threshold'])
        bid_accept_play_threshold = float(conf['sampling']['bid_accept_play_threshold'])
        sample_hands_auction = int(conf['sampling']['sample_hands_auction'])
        sample_boards_for_auction = int(conf['sampling']['sample_boards_for_auction'])
    
        sample_boards_for_auction_lead = int(conf['sampling']['sample_boards_for_auction_lead'])
        sample_hands_opening_lead = int(conf['sampling']['sample_hands_opening_lead'])
        sample_hands_play = int(conf['cardplay']['sample_hands_play'])

        return cls(lead_accept_threshold, bidding_threshold_sampling, play_accept_threshold, bid_accept_play_threshold, sample_hands_auction, sample_boards_for_auction, sample_boards_for_auction_lead, sample_hands_opening_lead, sample_hands_play, verbose)

    @property
    def sample_hands_auction(self):
        return self._sample_hands_auction
    
    def hand_to_str(self, hand):
        x = hand.reshape((4, 8))
        symbols = 'AKQJT98x'
        suits = []
        for i in range(4):
            s = ''
            for j in range(8):
                if x[i, j] > 0:
                    s += symbols[j] * x[i, j]
            suits.append(s)
        return '.'.join(suits)

    def sample_cards_vec(self, n_samples, p_hcp, p_shp, my_hand):
        #print(f"sample_cards_vec {my_hand}")
        deck = np.ones(32)
        deck[[7, 15, 23, 31]] = 6

        # unseen A K
        ak = np.zeros(32, dtype=int)
        ak[[0, 1, 8, 9, 16, 17, 24, 25]] = 1

        ak_out = ak - ak * my_hand
        ak_out_i_list = list(np.nonzero(ak_out)[0])
        ak_out_i = np.zeros((n_samples, len(ak_out_i_list)), dtype=int)
        ak_out_i[:, :] = np.array(ak_out_i_list)

        my_hand_small = my_hand * (1 - ak)

        small = deck * (1 - ak)

        small_out = small - my_hand_small
        small_out_i_list = get_small_out_i(small_out)
        small_out_i = np.zeros((n_samples, len(small_out_i_list)), dtype=int)
        small_out_i[:, :] = np.array(small_out_i_list)

        c_hcp = (lambda x: 4 * x + 10)(p_hcp.copy())
        c_shp = (lambda x: 1.75 * x + 3.25)(p_shp.copy()).reshape((3, 4))

        r_hcp = np.zeros((n_samples, 3)) + c_hcp
        r_shp = np.zeros((n_samples, 3, 4)) + c_shp

        lho_pard_rho = np.zeros((n_samples, 3, 32), dtype=int)
        cards_received = np.zeros((n_samples, 3), dtype=int)

        #print(ak_out_i)
        # all AK's in the same hand
        if (ak_out_i.shape[1] != 0): 
            #print(ak_out_i.shape[1])
            ak_out_i = np.vectorize(np.random.permutation,signature='(n)->(n)')(ak_out_i)
        small_out_i = np.vectorize(np.random.permutation, signature='(n)->(n)')(small_out_i)

        s_all = np.arange(n_samples)

        # distribute A and K
        js = np.zeros(n_samples, dtype=int)
        while np.min(js) < ak_out_i.shape[1]:
            cards = ak_out_i[s_all, js]
            receivers = distr2_vec(r_shp[s_all, :, cards//8], r_hcp)

            can_receive_cards = cards_received[s_all, receivers] < 13

            cards_received[s_all[can_receive_cards],
                           receivers[can_receive_cards]] += 1
            lho_pard_rho[s_all[can_receive_cards],
                         receivers[can_receive_cards], cards[can_receive_cards]] += 1
            r_hcp[s_all[can_receive_cards], receivers[can_receive_cards]] -= 3
            r_shp[s_all[can_receive_cards], receivers[can_receive_cards],
                  cards[can_receive_cards] // 8] -= 0.5
            js[can_receive_cards] += 1

        loop_counter = 0  # Initialize the loop counter
        # distribute small cards
        js = np.zeros(n_samples, dtype=int)
        while True:
            s_all_r = s_all[js < small_out_i.shape[1]]
            if len(s_all_r) == 0:
                break

            js_r = js[s_all_r]

            cards = small_out_i[s_all_r, js_r]
            receivers = distr_vec(r_shp[s_all_r, :, cards//8])

            can_receive_cards = cards_received[s_all_r, receivers] < 13

            cards_received[s_all_r[can_receive_cards],
                           receivers[can_receive_cards]] += 1
            lho_pard_rho[s_all_r[can_receive_cards],
                         receivers[can_receive_cards], cards[can_receive_cards]] += 1
            r_shp[s_all_r[can_receive_cards], receivers[can_receive_cards],
                  cards[can_receive_cards] // 8] -= 0.5
            js[s_all_r[can_receive_cards]] += 1

            loop_counter += 1  # Increment the loop counter
            if loop_counter >= 76:  # Check if the counter reaches 76
                break  #

        # re-apply constraints
        accept_hcp = np.ones(n_samples).astype(bool)

        for i in range(3):
            if np.round(c_hcp[i]) >= 11:
                accept_hcp &= binary.get_hcp(lho_pard_rho[:, i, :]) >= np.round(c_hcp[i]) - 2
            if np.round(c_hcp[i]) >= 15:
                accept_hcp &= binary.get_hcp(lho_pard_rho[:, i, :]) >= np.round(c_hcp[i]) - 1
            if np.round(c_hcp[i]) >= 18:
                accept_hcp &= binary.get_hcp(lho_pard_rho[:, i, :]) >= np.round(c_hcp[i])

        accept_shp = np.ones(n_samples).astype(bool)

        for i in range(3):
            for j in range(4):
                if np.round(c_shp[i, j] < 2):
                    accept_shp &= np.sum(lho_pard_rho[:, i, (j*8):((j+1)*8)], axis=1) <= np.round(c_shp[i, j]) + 1
                if np.round(c_shp[i, j] >= 4):
                    accept_shp &= np.sum(lho_pard_rho[:, i, (j*8):((j+1)*8)], axis=1) >= np.round(c_shp[i, j]) - 1
                if np.round(c_shp[i, j] >= 6):
                    accept_shp &= np.sum(lho_pard_rho[:, i, (j*8):((j+1)*8)], axis=1) >= np.round(c_shp[i, j])

        accept = accept_hcp & accept_shp

        if np.sum(accept) > 10:
            return lho_pard_rho[accept]
        else:
            return lho_pard_rho
        
    def sample_cards_auction(self, auction, nesw_i, hand, vuln, bidder_model, binfo, ns, ew, n_samples):

        if self.verbose:
            print("sample_cards_auction")
        n_steps = 1 + len(auction) // 4
         
        A = binary.get_auction_binary(n_steps, auction, nesw_i, hand, vuln, ns, ew)
        A_lho = binary.get_auction_binary(n_steps, auction, (nesw_i + 1) % 4, hand, vuln, ns, ew)
        A_pard = binary.get_auction_binary(n_steps, auction, (nesw_i + 2) % 4, hand, vuln, ns, ew)
        A_rho = binary.get_auction_binary(n_steps, auction, (nesw_i + 3) % 4, hand, vuln, ns, ew)

        p_hcp, p_shp = binfo.model(A)

        p_hcp = p_hcp.reshape((-1, n_steps, 3))[:, -1, :]
        p_shp = p_shp.reshape((-1, n_steps, 12))[:, -1, :]

        if self.verbose:
            c_hcp = (lambda x: 4 * x + 10)(p_hcp.copy())
            c_shp = (lambda x: 1.75 * x + 3.25)(p_shp.copy())
            #print(c_hcp[0])
            #print(c_shp[0])

        lho_pard_rho = self.sample_cards_vec(n_samples, p_hcp[0], p_shp[0], hand.reshape(32))

        n_samples = lho_pard_rho.shape[0]

        if self.verbose:
            print(f"n_samples {n_samples}")

        X_lho = np.zeros((n_samples, n_steps, A.shape[-1]))
        X_pard = np.zeros((n_samples, n_steps, A.shape[-1]))
        X_rho = np.zeros((n_samples, n_steps, A.shape[-1]))

        X_lho[:, :, :] = A_lho
        X_lho[:, :, 7:39] = lho_pard_rho[:, 0:1, :]
        X_lho[:, :, 2] = (binary.get_hcp(lho_pard_rho[:, 0, :]).reshape((-1, 1)) - 10) / 4
        X_lho[:, :, 3:7] = (binary.get_shape(lho_pard_rho[:, 0, :]).reshape((-1, 1, 4)) - 3.25) / 1.75
        lho_actual_bids = bidding.get_bid_ids(auction, (nesw_i + 1) % 4, n_steps)
        #print(lho_actual_bids)
        lho_sample_bids = bidder_model.model_seq(X_lho).reshape((n_samples, n_steps, -1))

        X_pard[:, :, :] = A_pard
        X_pard[:, :, 7:39] = lho_pard_rho[:, 1:2, :]
        X_pard[:, :, 2] = (binary.get_hcp(lho_pard_rho[:, 1, :]).reshape((-1, 1)) - 10) / 4
        X_pard[:, :, 3:7] = (binary.get_shape(lho_pard_rho[:, 1, :]).reshape((-1, 1, 4)) - 3.25) / 1.75
        pard_actual_bids = bidding.get_bid_ids(auction, (nesw_i + 2) % 4, n_steps)
        #print(pard_actual_bids)
        pard_sample_bids = bidder_model.model_seq(X_pard).reshape((n_samples, n_steps, -1))

        X_rho[:, :, :] = A_rho
        X_rho[:, :, 7:39] = lho_pard_rho[:, 2:, :]
        X_rho[:, :, 2] = (binary.get_hcp(lho_pard_rho[:, 2, :]).reshape((-1, 1)) - 10) / 4
        X_rho[:, :, 3:7] = (binary.get_shape(lho_pard_rho[:, 2, :]).reshape((-1, 1, 4)) - 3.25) / 1.75
        rho_actual_bids = bidding.get_bid_ids(auction, (nesw_i + 3) % 4, n_steps)
        #print(rho_actual_bids)
        rho_sample_bids = bidder_model.model_seq(X_rho).reshape((n_samples, n_steps, -1))
        #print(rho_sample_bids)
        min_scores = np.ones(n_samples)

        for i in range(n_steps):
            #print(bidding.ID2BID[lho_actual_bids[i]])
            #print(bidding.ID2BID[pard_actual_bids[i]])
            #print(bidding.ID2BID[rho_actual_bids[i]])
            if lho_actual_bids[i] not in (bidding.BID2ID['PAD_START'], bidding.BID2ID['PAD_END']):
                min_scores = np.minimum(min_scores, lho_sample_bids[:, i, lho_actual_bids[i]])
            if pard_actual_bids[i] not in (bidding.BID2ID['PAD_START'], bidding.BID2ID['PAD_END']):
                min_scores = np.minimum(min_scores, pard_sample_bids[:, i, pard_actual_bids[i]])
            if rho_actual_bids[i] not in (bidding.BID2ID['PAD_START'], bidding.BID2ID['PAD_END']):
                min_scores = np.minimum(min_scores, rho_sample_bids[:, i, rho_actual_bids[i]])

        # Get the indices that would sort min_scores in descending order
        sorted_indices = np.argsort(min_scores)[::-1]

        # Reorder the original lho_pard_rho array based on the sorted indices
        sorted_samples = lho_pard_rho[sorted_indices]
        # print(len(sorted_samples))

        # How much to trust the bidding for the samples
        accept_bidding_threshold = self.bidding_threshold_sampling
        accepted_samples = sorted_samples[min_scores > accept_bidding_threshold]
        # We sort the samples based on the score, so we get the hands matching the bidding best
        
        while accepted_samples.shape[0] < 50 and accept_bidding_threshold > 0.02:
            accept_bidding_threshold *= 0.9
            accepted_samples = sorted_samples[min_scores > accept_bidding_threshold]

        if len(accepted_samples) ==  0:
            # We found nothing that matches the bidding above the threshold of 0.02
            # Perhaps a longer bidding generally lowers the score for matching the bidding
            # For now we just return 3 best samples. That is better than none
            accepted_samples = sorted_samples[:3]

        return accepted_samples

    def shuffle_cards_bidding_info(self, n_samples, binfo, auction, hand, vuln, known_nesw, h_1_nesw, h_2_nesw, visible_cards, hidden_cards, cards_played, shown_out_suits, ns, ew):
        n_cards_to_receive = np.array(
            [len(hidden_cards) // 2, len(hidden_cards) - len(hidden_cards) // 2])

        n_steps = 1 + len(auction) // 4

        A = binary.get_auction_binary(
            n_steps, auction, known_nesw, hand, vuln, ns, ew)

        p_hcp, p_shp = binfo.model(A)

        p_hcp = p_hcp.reshape((-1, n_steps, 3))[:, -1, :]
        p_shp = p_shp.reshape((-1, n_steps, 12))[:, -1, :]

        def f_trans_hcp(x): return 4 * x + 10
        def f_trans_shp(x): return 1.75 * x + 3.25

        p_hcp = f_trans_hcp(
            p_hcp[0, [(h_1_nesw - known_nesw) % 4 - 1, (h_2_nesw - known_nesw) % 4 - 1]])
        p_shp = f_trans_shp(p_shp[0].reshape(
            (3, 4))[[(h_1_nesw - known_nesw) % 4 - 1, (h_2_nesw - known_nesw) % 4 - 1], :])

        h1_h2 = np.zeros((n_samples, 2, 32), dtype=int)
        cards_received = np.zeros((n_samples, 2), dtype=int)

        card_hcp = [4, 3, 2, 1, 0, 0, 0, 0] * 4

        # acknowledge all played cards
        for i, cards in enumerate(cards_played):
            for c in cards:
                p_hcp[i] -= card_hcp[c] / 1.2
                suit = c // 8
                p_shp[i, suit] -= 0.5

        # distribute all cards of suits which are known to have shown out
        cards_shownout_suits = []
        for i, suits in enumerate(shown_out_suits):
            for suit in suits:
                for card in filter(lambda x: x // 8 == suit, hidden_cards):
                    other_hand_i = (i + 1) % 2
                    h1_h2[:, other_hand_i, card] += 1
                    cards_received[:, other_hand_i] += 1
                    p_hcp[other_hand_i] -= card_hcp[card] / 1.2
                    p_shp[other_hand_i, suit] -= 0.5
                    cards_shownout_suits.append(card)

        hidden_cards = [
            c for c in hidden_cards if c not in cards_shownout_suits]
        ak_cards = [c for c in hidden_cards if c in {
            0, 1, 8, 9, 16, 17, 24, 25}]
        small_cards = [c for c in hidden_cards if c not in {
            0, 1, 8, 9, 16, 17, 24, 25}]

        ak_out_i = np.zeros((n_samples, len(ak_cards)), dtype=int)
        ak_out_i[:, :] = np.array(ak_cards)
        ak_out_i = np.vectorize(lambda x: np.random.permutation(np.copy(x)), signature='(n)->(n)')(ak_out_i)
        small_out_i = np.zeros((n_samples, len(small_cards)), dtype=int)
        small_out_i[:, :] = np.array(small_cards)
        small_out_i = np.vectorize(lambda x: np.random.permutation(np.copy(x)), signature='(n)->(n)')(small_out_i)

        r_hcp = np.zeros((n_samples, 2)) + p_hcp
        r_shp = np.zeros((n_samples, 2, 4)) + p_shp

        s_all = np.arange(n_samples)

        n_max_cards = np.zeros((n_samples, 2), dtype=int) + n_cards_to_receive

        js = np.zeros(n_samples, dtype=int)
        while True:
            s_all_r = s_all[js < ak_out_i.shape[1]]
            if len(s_all_r) == 0:
                break

            js_r = js[s_all_r]
            cards = ak_out_i[s_all_r, js_r]
            receivers = distr2_vec(r_shp[s_all_r, :, cards//8], r_hcp[s_all_r])

            can_receive_cards = cards_received[s_all_r,
                                               receivers] < n_max_cards[s_all_r, receivers]

            cards_received[s_all_r[can_receive_cards],
                           receivers[can_receive_cards]] += 1
            h1_h2[s_all_r[can_receive_cards],
                  receivers[can_receive_cards], cards[can_receive_cards]] += 1
            r_hcp[s_all_r[can_receive_cards],
                  receivers[can_receive_cards]] -= 3
            r_shp[s_all_r[can_receive_cards], receivers[can_receive_cards],
                  cards[can_receive_cards] // 8] -= 0.5
            js[s_all_r[can_receive_cards]] += 1

        js = np.zeros(n_samples, dtype=int)
        while True:
            s_all_r = s_all[js < small_out_i.shape[1]]
            if len(s_all_r) == 0:
                break

            js_r = js[s_all_r]
            cards = small_out_i[s_all_r, js_r]
            receivers = distr_vec(r_shp[s_all_r, :, cards//8])

            can_receive_cards = cards_received[s_all_r,
                                               receivers] < n_max_cards[s_all_r, receivers]

            cards_received[s_all_r[can_receive_cards],
                           receivers[can_receive_cards]] += 1
            h1_h2[s_all_r[can_receive_cards],
                  receivers[can_receive_cards], cards[can_receive_cards]] += 1
            r_shp[s_all_r[can_receive_cards], receivers[can_receive_cards],
                  cards[can_receive_cards] // 8] -= 0.5
            js[s_all_r[can_receive_cards]] += 1

        assert np.sum(h1_h2) == n_samples * np.sum(n_cards_to_receive)

        return h1_h2

    def get_opening_lead_scores(self, auction, vuln, binfo, lead_model, hand, opening_lead_card, ns, ew):
        contract = bidding.get_contract(auction)

        level = int(contract[0])
        strain = bidding.get_strain_i(contract)
        doubled = int('X' in contract)
        redbld = int('XX' in contract)

        x = np.zeros((hand.shape[0], 42))
        x[:, 0] = level
        x[:, 1 + strain] = 1
        x[:, 6] = doubled
        x[:, 7] = redbld

        decl_index = bidding.get_decl_i(contract)
        lead_index = (decl_index + 1) % 4

        vuln_us = vuln[lead_index % 2]
        vuln_them = vuln[decl_index % 2]

        x[:, 8] = vuln_us
        x[:, 9] = vuln_them
        x[:, 10:] = hand

        b = np.zeros((hand.shape[0], 15))

        n_steps = 1 + len(auction) // 4

        A = binary.get_auction_binary(
            n_steps, auction, lead_index, hand, vuln, ns, ew)

        p_hcp, p_shp = binfo.model(A)

        b[:, :3] = p_hcp.reshape((-1, n_steps, 3))[:, -1, :].reshape((-1, 3))
        b[:, 3:] = p_shp.reshape((-1, n_steps, 12))[:, -1, :].reshape((-1, 12))

        lead_softmax = lead_model.model(x, b)

        return lead_softmax[:, opening_lead_card]

    def get_bid_scores(self, nesw_i, auction, vuln, hand, bidder_model, ns, ew):
        n_steps = 1 + len(auction) // 4

        A = binary.get_auction_binary(
            n_steps, auction, nesw_i, hand, vuln, ns, ew)

        X = np.zeros((hand.shape[0], n_steps, A.shape[-1]))

        X[:, :, :2] = A[nesw_i, 0, :2]
        X[:, :, 7:39] = hand.reshape((-1, 1, 32))
        X[:, :, 39:] = A[nesw_i, :, 39:]
        X[:, :, 2] = (binary.get_hcp(hand).reshape((-1, 1)) - 10) / 4
        X[:, :, 3:7] = (binary.get_shape(
            hand).reshape((-1, 1, 4)) - 3.25) / 1.75
        actual_bids = bidding.get_bid_ids(auction, nesw_i, n_steps)
        sample_bids = bidder_model.model_seq(
            X).reshape((hand.shape[0], n_steps, -1))

        min_scores = np.ones(hand.shape[0])

        for i in range(n_steps):
            if actual_bids[i] not in (bidding.BID2ID['PAD_START'], bidding.BID2ID['PAD_END']):
                min_scores = np.minimum(
                    min_scores, sample_bids[:, i, actual_bids[i]])

        return min_scores

    def init_rollout_states(self, trick_i, player_i, card_players, player_cards_played, shown_out_suits, current_trick, auction, hand, vuln, models, ns, ew):
        n_samples = self.sample_hands_play
        if self.verbose:
            print(f"Called init_rollout_states {n_samples}")

        leader_i = (player_i - len(current_trick)) % 4
        hidden_1_i, hidden_2_i = [(3, 2), (0, 2), (0, 3), (2, 0)][player_i]

        # If no n_samples we are using cheat mode, where all cards are known
        if n_samples > 0:
            # sample the unknown cards
            public_hand_i = 3 if player_i == 1 else 1
            public_hand = card_players[public_hand_i].x_play[0, trick_i, :32]
            vis_cur_trick_nonpub = [c for i, c in enumerate(current_trick) if (leader_i + i) % 4 != public_hand_i]
            visible_cards = np.concatenate(
                [binary.get_cards_from_binary_hand(card_players[player_i].x_play[0, trick_i, :32]), binary.get_cards_from_binary_hand(public_hand)] +
                [np.array(vis_cur_trick_nonpub)] +
                [np.array(x, dtype=np.int32) for x in player_cards_played]
            )
            hidden_cards = get_all_hidden_cards(visible_cards)

            contract = bidding.get_contract(auction)
            known_nesw = player_to_nesw_i(player_i, contract)
            h_1_nesw = player_to_nesw_i(hidden_1_i, contract)
            h_2_nesw = player_to_nesw_i(hidden_2_i, contract)

            h1_h2 = self.shuffle_cards_bidding_info(
                40*n_samples,
                models.binfo,
                auction,
                hand,
                vuln,
                known_nesw,
                h_1_nesw,
                h_2_nesw,
                visible_cards,
                hidden_cards,
                [player_cards_played[hidden_1_i], player_cards_played[hidden_2_i]],
                [shown_out_suits[hidden_1_i], shown_out_suits[hidden_2_i]],
                ns,
                ew
            )

            hidden_hand1, hidden_hand2 = h1_h2[:, 0], h1_h2[:, 1]

            states = [np.zeros((hidden_hand1.shape[0], 13, 298)) for _ in range(4)]
            # we can reuse the x_play array from card_players except the player's hand
            for k in range(4):
                for i in range(trick_i + 1):
                    states[k][:, i, 32:] = card_players[k].x_play[0, i, 32:]

            # for player_i we can use the hand from card_players x_play (because the cards are known)
            for i in range(trick_i + 1):
                states[player_i][:, i, :32] = card_players[player_i].x_play[0, i, :32]

            # all players know dummy's cards
            if player_i in (0, 2, 3):
                for i in range(trick_i + 1):
                    states[player_i][:, i, 32:64] = card_players[1].x_play[0, i, :32]
                    states[1][:, i, :32] = card_players[1].x_play[0, i, :32]

            # dummy knows declarer's cards
            if player_i == 1:
                for i in range(trick_i + 1):
                    states[player_i][:, i, 32:64] = card_players[3].x_play[0, i, :32]
                    states[3][:, i, :32] = card_players[3].x_play[0, i, :32]
            # add the current trick cards to the hidden hands
            if len(current_trick) > 0:
                for i, card in enumerate(current_trick):
                    if (leader_i + i) % 4 == hidden_1_i:
                        hidden_hand1[:, card] += 1
                    if (leader_i + i) % 4 == hidden_2_i:
                        hidden_hand2[:, card] += 1

            for k in range(trick_i + 1):
                states[hidden_1_i][:, k, :32] = hidden_hand1
                states[hidden_2_i][:, k, :32] = hidden_hand2
                for card in player_cards_played[hidden_1_i][k:]:
                    states[hidden_1_i][:, k, card] += 1
                for card in player_cards_played[hidden_2_i][k:]:
                    states[hidden_2_i][:, k, card] += 1
            #print(states[0])
        else:
            # In cheat mode all cards are known
            contract = bidding.get_contract(auction)
            known_nesw = player_to_nesw_i(player_i, contract)
            states = [np.zeros((1, 13, 298)) for _ in range(4)]
            for k in range(4):
                for i in range(13):
                    states[k][0, i, :32] = card_players[k].x_play[0, i, :32]

        if self.verbose:
            print(f"players_states {states[0].shape[0]} trick {trick_i}")


        samples = []
        unique_indices = np.ones(states[0].shape[0]).astype(bool)
        for i in range(states[0].shape[0]):
            sample = '%s %s %s %s' % (
                self.hand_to_str(states[0][i,0,:32].astype(int)),
                self.hand_to_str(states[1][i,0,:32].astype(int)),
                self.hand_to_str(states[2][i,0,:32].astype(int)),
                self.hand_to_str(states[3][i,0,:32].astype(int)),
                )
            #print(sample)
            if sample in samples:
                unique_indices[i] = False
            else:
                samples.append(sample)
        # Use the unique_indices to filter player_states
        states = [state[unique_indices] for state in states]
        if self.verbose:
            print(f"Unique states {states[0].shape[0]}")
        if (states[0].shape[0] < n_samples // 2) or n_samples < 0:
            if self.verbose:
                print(f"Skipping re-apply constraints due to only {states[0].shape[0]} samples")
            return states

        # re-apply constraints
        n_steps = 1 + (len(auction)-1) // 4

        A = binary.get_auction_binary(n_steps, auction, known_nesw, hand, vuln, ns, ew)

        p_hcp, p_shp = models.binfo.model(A)

        # Only take the result from the latest bidding round
        p_hcp = p_hcp.reshape((-1, n_steps, 3))[:, -1, :]
        p_shp = p_shp.reshape((-1, n_steps, 12))[:, -1, :]

        def f_trans_hcp(x): return 4 * x + 10
        def f_trans_shp(x): return 1.75 * x + 3.25

        p_hcp = f_trans_hcp(p_hcp[0, [(h_1_nesw - known_nesw) % 4 - 1, (h_2_nesw - known_nesw) % 4 - 1]])
        p_shp = f_trans_shp(p_shp[0].reshape((3, 4))[[(h_1_nesw - known_nesw) % 4 - 1, (h_2_nesw - known_nesw) % 4 - 1], :])

        c_hcp = p_hcp.copy()
        c_shp = p_shp.copy()

        if self.verbose:
            print(f"c_hcp:{c_hcp}")
            print(f"c_shp:{c_shp}")

        accept_hcp = np.ones(states[0].shape[0]).astype(bool)

        for i in range(2):
            if np.round(c_hcp[i]) >= 11:
                accept_hcp &= binary.get_hcp(states[[hidden_1_i, hidden_2_i][i]][:, 0, :32]) >= np.round(c_hcp[i]) - 2
            if np.round(c_hcp[i]) >= 15:
                accept_hcp &= binary.get_hcp(states[[hidden_1_i, hidden_2_i][i]][:, 0, :32]) >= np.round(c_hcp[i]) - 1
            if np.round(c_hcp[i]) >= 18:
                accept_hcp &= binary.get_hcp(states[[hidden_1_i, hidden_2_i][i]][:, 0, :32]) >= np.round(c_hcp[i])

        accept_shp = np.ones(states[0].shape[0]).astype(bool)

        for i in range(2):
            for j in range(4):
                if np.round(c_shp[i, j] < 2):
                    accept_shp &= np.sum(states[[hidden_1_i, hidden_2_i][i]][:, 0, (j*8):((j+1)*8)], axis=1) <= np.round(c_shp[i, j]) + 1
                if np.round(c_shp[i, j] >= 5):
                    accept_shp &= np.sum(states[[hidden_1_i, hidden_2_i][i]][:, 0, (j*8):((j+1)*8)], axis=1) >= np.round(c_shp[i, j]) - 1
                if np.round(c_shp[i, j] >= 6):
                    accept_shp &= np.sum(states[[hidden_1_i, hidden_2_i][i]][:, 0, (j*8):((j+1)*8)], axis=1) >= np.round(c_shp[i, j])

        accept = accept_hcp & accept_shp

        if np.sum(accept) < n_samples:
            accept = np.ones_like(accept).astype(bool)
        # end of re-applyconstraints

        states = [state[accept] for state in states]
        if self.verbose:   
            print(f"States {states[0].shape[0]} before opening lead (after shape and hcp)")
        if (states[0].shape[0] < n_samples // 2 or n_samples < 10):
            #print(f"Skipping re-apply constraints due to only {states[0].shape[0]} samples")
            return states

        # reject samples inconsistent with the opening lead
        # t_start = time.time()
        if hidden_1_i == 0 or hidden_2_i == 0:
            opening_lead = current_trick[0] if trick_i == 0 else player_cards_played[0][0]
            lead_scores = self.get_opening_lead_scores(
                auction, vuln, models.binfo, models.lead, states[0][:, 0, :32], opening_lead, ns, ew)

            # How much trust that opponents would have lead the actual card from the hand sampled
            lead_accept_threshold = self.lead_accept_threshold
            while np.sum(lead_scores > lead_accept_threshold) < 20:
                lead_accept_threshold *= 0.9

            states = [state[lead_scores > lead_accept_threshold]
                      for state in states]
            #print(f"States {states[0].shape[0]} after cheking lead")
            if (states[0].shape[0] < n_samples // 2):
                #print(f"Skipping re-apply constraints due to only {states[0].shape[0]} samples")
                return states

        # reject samples inconsistent with the bidding
        if trick_i <= 9:
            # start = time.time()
            min_bid_scores = np.ones(states[0].shape[0])

            for h_i in [hidden_1_i, hidden_2_i]:
                h_i_nesw = player_to_nesw_i(h_i, contract)
                #print(f"h_i = {h_i} h_i_nesw = {h_i_nesw}")
                bid_scores = self.get_bid_scores(
                    h_i_nesw, auction, vuln, states[h_i][:, 0, :32], models.bidder_model, ns, ew)
                #indices_of_highest_values = np.argsort(bid_scores)[:10][::-1]
                #print(indices_of_highest_values)
                #for index in indices_of_highest_values:
                #    print(f"Index: {index}, Value: {bid_scores[index]}")
                min_bid_scores = np.minimum(min_bid_scores, bid_scores)

            # trusting the bidding after sampling cards
            # This could probably be set based on number of deals matching or sorted
            bid_accept_threshold = self.bid_accept_play_threshold

            while np.sum(min_bid_scores > bid_accept_threshold) < 20:
                bid_accept_threshold *= 0.9

            states = [state[min_bid_scores > bid_accept_threshold]
                      for state in states]
            #print(f"States {states[0].shape[0]} after checking the bidding")
            if (states[0].shape[0] < n_samples // 2):
                if self.verbose:
                    print(f"Skipping re-apply constraints due to only {states[0].shape[0]} samples")
                return states

        # To save time we reduce no of samples to 2 times what is required
        # states = [state[:2*n_samples] for state in states]

        # reject samples inconsistent with the play
        # t_start = time.time()
        min_scores = np.ones(states[0].shape[0])

        if trick_i <= 9:
            for p_i in [hidden_1_i, hidden_2_i]:

                if trick_i == 0 and p_i == 0:
                    continue

                card_played_current_trick = []
                for i, card in enumerate(current_trick):
                    if (leader_i + i) % 4 == p_i:
                        card_played_current_trick.append(card)

                cards_played_prev_tricks = player_cards_played[p_i][1:
                                                                    trick_i] if p_i == 0 else player_cards_played[p_i][:trick_i]

                cards_played = cards_played_prev_tricks + card_played_current_trick

                if len(cards_played) == 0:
                    continue

                n_tricks_pred = max(
                    10, trick_i + len(card_played_current_trick))
                p_cards = models.player_models[p_i].model(
                    states[p_i][:, :n_tricks_pred, :])

                card_scores = p_cards[:, np.arange(
                    len(cards_played)), cards_played]

                min_scores = np.minimum(
                    min_scores, np.min(card_scores, axis=1))

        # Trust in the play until now
        play_accept_threshold = self.play_accept_threshold

        while np.sum(min_scores > play_accept_threshold) < 20 and play_accept_threshold > 0:
            play_accept_threshold -= 0.01
            #print(f"play_accept_threshold {play_accept_threshold} reduced")

        s_accepted = min_scores > play_accept_threshold

        states = [state[s_accepted] for state in states]
        if self.verbose:
            print(f"States {states[0].shape[0]} after checking the play")
        return [state[:n_samples] for state in states]
