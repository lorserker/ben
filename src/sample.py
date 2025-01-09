import sys
import time

import numpy as np
import tensorflow as tf

import binary

from bidding import bidding
import deck52
from util import get_all_hidden_cards, calculate_seed, convert_to_probability, convert_to_probability_with_weight
from configparser import ConfigParser
from util import hand_to_str
from collections import defaultdict
from colorama import Fore, Back, Style, init

init()

np.set_printoptions(precision=3, suppress=True, linewidth=220, threshold=np.inf)

def get_small_out_i(small_out):
    x = small_out.copy()
    dec = np.minimum(1, x)

    result = []
    while np.max(x) > 0:
        result.extend(np.nonzero(x)[0])
        x = x - dec
        dec = np.minimum(1, x)

    return result


def distr_vec(x, rng):
    xpos = np.maximum(x, 0) + 0.1
    pvals = xpos / np.sum(xpos, axis=1, keepdims=True)

    p_cumul = np.cumsum(pvals, axis=1)

    indexes = np.zeros(pvals.shape[0], dtype=np.int32)
    rnd = rng.random(pvals.shape[0])

    for k in range(p_cumul.shape[1]):
        indexes = indexes + (rnd > p_cumul[:, k])

    return indexes


def distr2_vec(x1, x2, rng):
    x1pos = np.maximum(x1, 0) + 0.1
    x2pos = np.maximum(x2, 0) + 0.1

    pvals1 = x1pos / np.sum(x1pos, axis=1, keepdims=True)
    pvals2 = x2pos / np.sum(x2pos, axis=1, keepdims=True)

    pvals = pvals1 * pvals2
    pvals = pvals / np.sum(pvals, axis=1, keepdims=True)

    return distr_vec(pvals,rng)


def player_to_nesw_i(player_i, contract):
    decl_i = bidding.get_decl_i(contract)
    return (decl_i + player_i + 1) % 4

class Sample:

    def __init__(self, lead_accept_threshold, lead_accept_threshold_suit, lead_accept_threshold_honors, lead_accept_threshold_partner_trust, bidding_threshold_sampling, play_accept_threshold, play_accept_threshold_partner, min_play_accept_threshold_samples, bid_accept_play_threshold, 
                 bid_accept_threshold_bidding, bid_extend_play_threshold, sample_hands_auction, min_sample_hands_auction, sample_boards_for_auction, sample_boards_for_auction_step, warn_to_few_samples, increase_for_bid_count, sample_boards_for_auction_opening_lead, 
                 sample_hands_opening_lead, sample_hands_play, min_sample_hands_play, min_sample_hands_play_bad, sample_boards_for_play, use_biddinginfo, use_distance, no_samples_when_no_search, exclude_samples, no_biddingqualitycheck_after_bid_count, 
                 hcp_reduction_factor, shp_reduction_factor, sample_previous_round_if_needed, verbose):
        self.lead_accept_threshold = lead_accept_threshold
        self.lead_accept_threshold_suit = lead_accept_threshold_suit
        self.lead_accept_threshold_honors = lead_accept_threshold_honors
        self.lead_accept_threshold_partner_trust = lead_accept_threshold_partner_trust
        self.bidding_threshold_sampling = bidding_threshold_sampling
        self.play_accept_threshold = play_accept_threshold
        self.play_accept_threshold_partner = play_accept_threshold_partner
        self.min_play_accept_threshold_samples = min_play_accept_threshold_samples
        self.bid_accept_play_threshold = bid_accept_play_threshold
        self.bid_accept_threshold_bidding = bid_accept_threshold_bidding
        self.bid_extend_play_threshold = bid_extend_play_threshold
        self._sample_hands_auction = sample_hands_auction
        self._min_sample_hands_auction = min_sample_hands_auction
        self.sample_boards_for_auction = sample_boards_for_auction
        self.sample_boards_for_auction_step = sample_boards_for_auction_step
        self.warn_to_few_samples = warn_to_few_samples
        self.increase_for_bid_count = increase_for_bid_count
        self.sample_boards_for_auction_opening_lead = sample_boards_for_auction_opening_lead
        self.sample_hands_opening_lead = sample_hands_opening_lead
        self.sample_hands_play = sample_hands_play
        self.min_sample_hands_play = min_sample_hands_play
        self.min_sample_hands_play_bad = min_sample_hands_play_bad
        self.sample_boards_for_play = sample_boards_for_play
        self.use_biddinginfo = use_biddinginfo
        self.use_distance = use_distance
        self.no_samples_when_no_search = no_samples_when_no_search
        self.exclude_samples = exclude_samples
        self.no_biddingqualitycheck_after_bid_count = no_biddingqualitycheck_after_bid_count
        self.hcp_reduction_factor = hcp_reduction_factor
        self.shp_reduction_factor = shp_reduction_factor
        self.sample_previous_round_if_needed = sample_previous_round_if_needed
        self.verbose = verbose


    @classmethod
    def from_conf(cls, conf: ConfigParser, verbose=False) -> "Sample":
        lead_accept_threshold = float(conf['sampling']['lead_accept_threshold'])
        lead_accept_threshold_suit = conf.getboolean('sampling', 'lead_accept_threshold_suit', fallback=False)
        lead_accept_threshold_honors = conf.getboolean('sampling', 'lead_accept_threshold_honors', fallback=False)
        lead_accept_threshold_partner_trust = float(conf['sampling']['lead_accept_threshold_partner_trust'])
        bidding_threshold_sampling = float(conf['sampling']['bidding_threshold_sampling'])
        play_accept_threshold = float(conf['sampling']['play_accept_threshold'])
        play_accept_threshold_partner = float(conf['sampling'].get('play_accept_threshold_partner', play_accept_threshold))
        min_play_accept_threshold_samples = conf.getint('sampling','min_play_accept_threshold_samples',fallback=20)
        bid_accept_play_threshold = float(conf['sampling']['bid_accept_play_threshold'])
        bid_accept_threshold_bidding = float(conf['sampling']['bid_accept_threshold_bidding'])
        bid_extend_play_threshold = float(conf['sampling'].get('bid_extend_play_threshold', 0))
        exclude_samples = float(conf['sampling'].get('exclude_samples', 0))
        sample_hands_auction = int(conf['sampling']['sample_hands_auction'])
        min_sample_hands_auction = int(conf['sampling']['min_sample_hands_auction'])
        sample_boards_for_auction = int(conf['sampling']['sample_boards_for_auction'])
        sample_boards_for_auction = conf.getint('sampling','sample_boards_for_auction',fallback=2000)
        sample_boards_for_auction_step = conf.getint('sampling','sample_boards_for_auction_step',fallback=1000)
        warn_to_few_samples = conf.getint('sampling','warn_to_few_samples',fallback=10)
        increase_for_bid_count = conf.getint('sampling','increase_for_bid_count',fallback=6)
        sample_boards_for_auction_opening_lead = int(conf['sampling']['sample_boards_for_auction_opening_lead'])
        sample_hands_opening_lead = int(conf['sampling']['sample_hands_opening_lead'])
        sample_hands_play = int(conf['cardplay']['sample_hands_play'])
        min_sample_hands_play = int(conf['cardplay']['min_sample_hands_play'])
        min_sample_hands_play_bad = conf.getint('cardplay','min_sample_hands_play_bad',fallback=3)
        sample_boards_for_play = int(conf['cardplay']['sample_boards_for_play'])
        use_biddinginfo = conf.getboolean('cardplay', 'use_biddinginfo', fallback=True)
        use_distance = conf.getboolean('sampling', 'use_distance', fallback=False)
        no_samples_when_no_search = conf.getboolean('sampling', 'no_samples_when_no_search', fallback=False)
        no_biddingquality_after_bid_count = conf.getint('bidding', 'no_biddingquality_after_bid_count', fallback=12)
        hcp_reduction_factor = conf.getfloat('sampling', 'hcp_reduction_factor', fallback=0.9)
        shp_reduction_factor = conf.getfloat('sampling', 'shp_reduction_factor', fallback=0.5)
        sample_previous_round_if_needed = conf.getboolean('sampling', 'sample_previous_round_if_needed', fallback=False)        
        return cls(lead_accept_threshold, lead_accept_threshold_suit, lead_accept_threshold_honors, lead_accept_threshold_partner_trust, bidding_threshold_sampling, play_accept_threshold, play_accept_threshold_partner, min_play_accept_threshold_samples, 
                   bid_accept_play_threshold, bid_accept_threshold_bidding, bid_extend_play_threshold, sample_hands_auction, min_sample_hands_auction, sample_boards_for_auction, sample_boards_for_auction_step, warn_to_few_samples, increase_for_bid_count, 
                   sample_boards_for_auction_opening_lead, sample_hands_opening_lead, sample_hands_play, min_sample_hands_play, min_sample_hands_play_bad, sample_boards_for_play, use_biddinginfo, use_distance, no_samples_when_no_search, 
                   exclude_samples, no_biddingquality_after_bid_count, hcp_reduction_factor, shp_reduction_factor, sample_previous_round_if_needed, verbose)

    @property
    def sample_hands_auction(self):
        return self._sample_hands_auction

    @property
    def min_sample_hands_auction(self):
        return self._min_sample_hands_auction

    def generate_samples_iterative(self, auction_so_far, turn_to_bid, max_samples, needed_samples, rng, hand_str, vuln, models, accepted_samples):
        # The longer the aution the more hands we might need to sample
        sample_boards_for_auction = max_samples
        
        bid_count = binary.get_number_of_bids_without_pass(auction_so_far)
        if self.increase_for_bid_count > 0:
            factor = bid_count // self.increase_for_bid_count
            if factor > 0:
                sample_boards_for_auction = round((1.5 ** factor) * sample_boards_for_auction)
                if self.verbose:
                    print(f"Increasing samples to {sample_boards_for_auction} for auction with {binary.get_number_of_bids(auction_so_far)} bids")


        sorted_scores = []
        samplings = 0
        current_count = len(accepted_samples)
        quality = 0
        p_hcp, p_shp = None, None
        step = self.sample_boards_for_auction_step
        extend_samples = False
        while samplings < sample_boards_for_auction and current_count <= needed_samples:
            new_samples, new_sorted_scores, p_hcp, p_shp, new_quality = self.sample_cards_auction(auction_so_far, turn_to_bid, hand_str, vuln, step, rng, models, p_hcp, p_shp, extend_samples, self.verbose and current_count == 0)
            current_count += new_samples.shape[0]
            quality += new_quality * new_samples.shape[0]
            # Accumulate the samples and scores
            # We should probably check for duplicates
            accepted_samples.append(new_samples)
            sorted_scores.append(new_sorted_scores)
            samplings += step
            extend_samples = (samplings + 3*step >= max_samples)
            # In the first iterations we did not find enough samples
            # so this time we are going to extend below threshold, and create a large sample
            # Another option could be to save doubted samples from earlier iterations
            if current_count < needed_samples and extend_samples:
                step = self.sample_boards_for_auction_step * 3

        if current_count == 0:
            sys.stderr.write(f"{Fore.RED}No samples found for auction {auction_so_far} - Samplings: {samplings} max {sample_boards_for_auction}{Fore.RESET}\n")

        # Convert list of arrays into a single array along the first dimension
        quality = -1 if current_count == 0 else quality / current_count
        accepted_samples = np.concatenate(accepted_samples, axis=0)
        sorted_scores = np.concatenate(sorted_scores, axis=0)
        if self.verbose:
            print(f"Found {len(accepted_samples)} samples for bidding. Quality={quality:.2f}, Samplings={samplings}, Auction={auction_so_far}")
        
        if self.sample_previous_round_if_needed and len(accepted_samples) < self._min_sample_hands_auction and len(auction_so_far) >= 8:
            sys.stderr.write(f"{Fore.YELLOW}Quality {quality:.2f} to low for auction {auction_so_far} - Samplings: {samplings} max {sample_boards_for_auction} - Skipping last bidding round{Fore.RESET}\n")
            auction_so_far_copy = auction_so_far.copy()
            auction_so_far_copy = auction_so_far_copy[:-4]
            needed_samples = needed_samples / 4
            # Consider transferring found samples, but we also need score and quality then
            accepted_samples = []
            return self.generate_samples_iterative(auction_so_far_copy, turn_to_bid, max_samples, needed_samples, rng, hand_str, vuln, models, accepted_samples)


        if self.sample_previous_round_if_needed and quality < self.bid_accept_threshold_bidding and len(auction_so_far) >= 8:
            sys.stderr.write(f"{Fore.YELLOW}Quality {quality:.2f} to low for auction {auction_so_far} - Samplings: {samplings} max {sample_boards_for_auction} - Skipping their doubles{Fore.RESET}\n")
            # Was there a X or XX we can replace with P, then just try again
            auction_updated = False
            auction_so_far_copy = auction_so_far.copy()
            # Replace every second 'X' starting from the last
            count = 0
            for i in range(len(auction_so_far) - 1, -1, -1):  # Iterate backwards
                if auction_so_far[i] == 'X':
                    count += 1
                    if count % 2 == 1:  # Replace only every second 'X'
                        auction_updated = True
                        auction_so_far_copy[i] = 'PASS'
                else:
                    if auction_so_far_copy[i] != 'PASS':
                        break
            # It was a double in last bidding round, and changing that to pass will end the bidding
            if not bidding.auction_over(auction_so_far_copy):
                if auction_updated:
                    sys.stderr.write(f"{Fore.YELLOW}Updated auction {auction_so_far_copy}{Fore.RESET}\n")
                    needed_samples = needed_samples / 4
                    # Consider transferring found samples, but we also need score and quality then
                    accepted_samples = []
                    return self.generate_samples_iterative(auction_so_far_copy, turn_to_bid, max_samples, needed_samples, rng, hand_str, vuln, models, accepted_samples)
                else:
                    sys.stderr.write(f"{Fore.YELLOW}Could not update auction {auction_so_far}{Fore.RESET}\n")

        return accepted_samples, sorted_scores, p_hcp, p_shp, quality, samplings

    def sample_cards_vec(self, n_samples, c_hcp, c_shp, my_hand, rng, n_cards=32):
        if self.verbose:
            print("sample_cards_vec generating", n_samples, "Seed:", rng.bit_generator.state['state']['state'])
            t_start = time.time()
        deck = np.ones(n_cards, dtype=int)
        cards_in_suit = n_cards // 4

        for j in range(4):
            deck[cards_in_suit* (j  + 1) -1] = 14 - cards_in_suit
        
        #for i in range(4):
        #    deck[[7, 15, 23, 31]] = 1 + 13 - cards_in_suit

        # unseen A K
        ak = np.zeros(n_cards, dtype=int)
        #for i in range(4):
        #    ak[[0, 1, 8, 9, 16, 17, 24, 25]] = 1

        # Generate indices with the specified increments
        indices = [i for j in range(4) for i in ( j * cards_in_suit, j * cards_in_suit + 1)]

        # Assign values to the specified indices
        ak[indices] = 1

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

        # calculate missing hcp
        missing_hcp = 40 - binary.get_hcp(np.array([my_hand]))[0]

        if self.use_biddinginfo:
            #c_hcp = (lambda x: 4 * x + 10)(p_hcp.copy())
            c_shp = c_shp.reshape((3, 4))
            r_hcp = np.zeros((n_samples, 3)) + c_hcp
            r_shp = np.zeros((n_samples, 3, 4)) + c_shp

            if missing_hcp > 0:
                hcp_reduction_factor = round(self.hcp_reduction_factor * np.sum(r_hcp[0]) / missing_hcp,2)
            else:
                hcp_reduction_factor = 0

            shp_reduction_factor = self.shp_reduction_factor
        else:
            r_hcp = np.ones((n_samples, 3)) 
            r_shp = np.ones((n_samples, 3, 4))
            hcp_reduction_factor = 0
            shp_reduction_factor = 0

        lho_pard_rho = np.zeros((n_samples, 3, n_cards), dtype=int)
        cards_received = np.zeros((n_samples, 3), dtype=int)


        if self.verbose:
            print("Missing HCP   :", missing_hcp,"Expected HCP  :",r_hcp[0],"Expected Shape:","".join(map(str, r_shp[0])),
                  f"hcp_reduction_factor:{hcp_reduction_factor}  {self.hcp_reduction_factor}",
                  f"shp_reduction_factor:{shp_reduction_factor}  {self.shp_reduction_factor}")            

        # all AK's in the same hand
        # vectorize has an overhead
        if (ak_out_i.shape[1] != 0):
            # print(ak_out_i.shape[1])
            ak_out_i = np.vectorize(rng.permutation, signature='(n)->(n)')(ak_out_i)
        small_out_i = np.vectorize(rng.permutation, signature='(n)->(n)')(small_out_i)

        s_all = np.arange(n_samples)

        # distribute AK
        js = np.zeros(n_samples, dtype=int)
        while np.min(js) < ak_out_i.shape[1]:
            cards = ak_out_i[s_all, js]
            receivers = distr2_vec(r_shp[s_all, :, cards//cards_in_suit], r_hcp, rng)

            can_receive_cards = cards_received[s_all, receivers] < 13

            cards_received[s_all[can_receive_cards], receivers[can_receive_cards]] += 1
            lho_pard_rho[s_all[can_receive_cards], receivers[can_receive_cards], cards[can_receive_cards]] += 1
            if self.use_biddinginfo:
                r_hcp[s_all[can_receive_cards], receivers[can_receive_cards]] -= 3 * hcp_reduction_factor
                r_shp[s_all[can_receive_cards], receivers[can_receive_cards], cards[can_receive_cards] // cards_in_suit] -= self.shp_reduction_factor
            js[can_receive_cards] += 1

        # distribute small cards
        js = np.zeros(n_samples, dtype=int)
        loop_counter = 0
        while True:
            s_all_r = s_all[js < small_out_i.shape[1]]
            if len(s_all_r) == 0:
                break

            js_r = js[s_all_r]

            cards = small_out_i[s_all_r, js_r]
            receivers = distr_vec(r_shp[s_all_r, :, cards//cards_in_suit], rng)

            can_receive_cards = cards_received[s_all_r, receivers] < 13

            cards_received[s_all_r[can_receive_cards], receivers[can_receive_cards]] += 1
            lho_pard_rho[s_all_r[can_receive_cards], receivers[can_receive_cards], cards[can_receive_cards]] += 1
            r_shp[s_all_r[can_receive_cards], receivers[can_receive_cards], cards[can_receive_cards] // cards_in_suit] -= self.shp_reduction_factor
            js[s_all_r[can_receive_cards]] += 1

            # This loop_counter stops the handgeneration, and might be implemented to stop instead of using time to get last cards distributed
            # This can result in hands with 12 cards, and interestingly the bidding can be OK with 12 cards, and also single dummy is fine
            # But after implementing the option of Double Dummy for opening lead it is a problem
            # So we could just ship the boards where a player has 12 cards, but for now we just remove the counter
            loop_counter += 1  # Increment the loop counter
            #if loop_counter >= 250:  # Check if the counter reaches 76
            #    print("Loop counter >= 76")
            #    break  #

        #if self.verbose:
        #    print("Loops to deal the hands", loop_counter)

        if self.use_biddinginfo:

            # re-apply constraints
            # This is in principle just to reduce the number of samples for performance
            accept_hcp = np.ones(n_samples).astype(bool)
            
            for i in range(3):
                if np.round(c_hcp[i]) >= 11:
                    accept_hcp &= binary.get_hcp(lho_pard_rho[:, i, :]) >= np.round(c_hcp[i]) - 5

            #if self.verbose:
            #    print(f'sample_cards_vec took {(time.time() - t_start):0.4f} Deals hcp accepted: {np.sum(accept_hcp)}')

            accept_shp = np.ones(n_samples).astype(bool)

            for i in range(3):
                for j in range(4):
                    if np.round(c_shp[i, j] < 2):
                        accept_shp &= np.sum(lho_pard_rho[:, i, (j*cards_in_suit):((j+1)*cards_in_suit)], axis=1) <= np.round(c_shp[i, j]) + 1
                    if np.round(c_shp[i, j] >= 6):
                        accept_shp &= np.sum(lho_pard_rho[:, i, (j*cards_in_suit):((j+1)*cards_in_suit)], axis=1) >= np.round(c_shp[i, j]) - 1

            #if self.verbose:
            #    print(f'sample_cards_vec took {(time.time() - t_start):0.4f} Deals shape accepted: {np.sum(accept_shp)}')

            accept = accept_hcp & accept_shp

            accepted = np.sum(accept)
            if self.verbose:
                print(f'sample_cards_vec took {(time.time() - t_start):0.4f} Deals: {accepted}')
            # If we have filtered to many away just return all samples - performance            
            if accepted >= n_samples / 2:
                return lho_pard_rho[accept]
            else:
                return lho_pard_rho
        else:
            return lho_pard_rho


    def get_bidding_info(self, n_steps, auction, nesw_i, hand, vuln, models):
        assert n_steps > 0, "n_steps should be greater than zero"
        A = binary.get_auction_binary_sampling(n_steps, auction, nesw_i, hand, vuln, models, models.n_cards_bidding)
        p_hcp, p_shp = models.binfo_model.pred_fun(A)
        if tf.is_tensor(p_hcp):
            p_hcp = p_hcp.numpy()
            p_shp = p_shp.numpy()

        p_hcp = p_hcp.reshape((-1, n_steps, 3))[:, -1, :]
        p_shp = p_shp.reshape((-1, n_steps, 12))[:, -1, :]

        c_hcp = (lambda x: 4 * x + 10)(p_hcp.copy())
        c_shp = (lambda x: 1.75 * x + 3.25)(p_shp.copy())

        if self.verbose:
            print("Player: ", 'NESW'[nesw_i], "Hand: ", hand_to_str(hand[0],models.n_cards_bidding))
            print("HCP: ", c_hcp)
            print("Shape: ", c_shp)

        return c_hcp, c_shp
    
    def process_bidding(self, player, lho_pard_rho, min_scores_lho, min_scores_partner, min_scores_rho, 
                    auction, nesw_i, hand, vuln, models, index, actual_bids, n_steps, size, model, verbose):
        position = player["position"]
        bid_count = player["bid_count"]
        actual_bids = player["actual_bids"]        
        if bid_count > 0:
            X = np.zeros((lho_pard_rho.shape[0], n_steps, size), dtype=np.float16)
            A = binary.get_auction_binary_sampling(n_steps, auction, (nesw_i + position) % 4, hand, vuln, models, models.n_cards_bidding)
            X[:, :, :] = A
            X[:, :, 7+index:7+models.n_cards_bidding+index] = lho_pard_rho[:, position-1:position, :]
            X[:, :, 2+index] = (binary.get_hcp(lho_pard_rho[:, position-1, :]).reshape((-1, 1)) - 10) / 4
            X[:, :, 3+index:7+index] = (binary.get_shape(lho_pard_rho[:, position-1, :]).reshape((-1, 1, 4)) - 3.25) / 1.75

            # Predict bids based on the model for the specific position
            sample_bids, _ = model.pred_fun_seq(X)
            # Check if sample_bids is a TensorFlow tensor
            if tf.is_tensor(sample_bids):
                sample_bids = sample_bids.numpy()
            sample_bids = sample_bids.reshape((lho_pard_rho.shape[0], n_steps, -1))

            # Update min scores based on actual bids
            min_scores = np.ones_like(min_scores_lho)
            for i in range(n_steps):
                if actual_bids[i] not in (bidding.BID2ID['PAD_START'], bidding.BID2ID['PAD_END']):
                    min_scores = np.minimum(min_scores, sample_bids[:, i, actual_bids[i]])

            # Filter samples and update all min scores arrays based on `exclude_samples`
            mask = min_scores >= self.exclude_samples
            lho_pard_rho = lho_pard_rho[mask]
            min_scores_lho = min_scores_lho[mask]
            min_scores_partner = min_scores_partner[mask]
            min_scores_rho = min_scores_rho[mask]
            # Now update the min scores based on response from the NN
            if position == 1:
                min_scores_lho = min_scores[mask]
            if position == 2:
                min_scores_partner = min_scores[mask]
            if position == 3:
                min_scores_rho = min_scores[mask]
            
        else:
            if verbose:
                print(f"No bidding for position {position}")
        return lho_pard_rho, min_scores_lho, min_scores_partner, min_scores_rho

        
    def sample_cards_auction(self, auction, nesw_i, hand_str, vuln, n_samples, rng, models, old_c_hcp, old_c_shp, extend_samples = True, verbose = False):
        hand = binary.parse_hand_f(models.n_cards_bidding)(hand_str)
        n_steps = binary.calculate_step_bidding_info(auction)
        bids = 4 if models.model_version >= 2 else 3
        if verbose:
            print("sample_cards_auction, nsteps=", n_steps)
            print("NS: ", models.ns, "EW: ", models.ew, "Auction: ", auction)
            print("hand", hand_str)
            print("nesw_i", nesw_i)
            print("n_samples", n_samples)
        
        if old_c_shp is None:            
            c_hcp, c_shp = self.get_bidding_info(n_steps, auction, nesw_i, hand, vuln, models)
            c_hcp = c_hcp[0]
            c_shp = c_shp[0]
        else:
            c_hcp = old_c_hcp
            c_shp = old_c_shp

        lho_pard_rho = self.sample_cards_vec(n_samples, c_hcp, c_shp, hand.reshape(models.n_cards_bidding), rng, models.n_cards_bidding)

        # Consider saving the generated boards, and add the result from previous sampling to this output
        n_samples = lho_pard_rho.shape[0]
        if verbose:
            print(f"n_samples {n_samples} from bidding info")

        n_steps = binary.calculate_step_bidding(auction)

        #print(auction)

        index = 0 if models.model_version == 0 or models.ns == -1 else 2
            
        size = 7 + models.n_cards_bidding + index + bids*40
        
        # Initialize the players list with a loop to get bid information for each position
        players = []
        for i in range(1, 4):
            actual_bids, bid_count = bidding.get_bid_ids(auction, (nesw_i + i) % 4, n_steps)
            players.append({
                "position": i,
                "bid_count": bid_count,
                "actual_bids": actual_bids
            })
        lho_bid_count = players[0]["bid_count"]
        pard_bid_count = players[1]["bid_count"]
        rho_bid_count = players[2]["bid_count"]
        

        # Calculate the total bid_count for all players
        total_bid_count = sum(player["bid_count"] for player in players)
        if total_bid_count == 0:
            # No bidding yet, so we just return the first samples
            quality = 1
            sorted_scores = np.ones(self.sample_hands_auction)
            accepted_samples = lho_pard_rho[:self.sample_hands_auction]
            return accepted_samples, sorted_scores, c_hcp, c_shp, quality


        # Sort players by bid_count in descending order
        #players.sort(key=lambda x: x["bid_count"], reverse=True)
        # Rearrange the players in the order [2, 3, 1]
        players = [players[1], players[2], players[0]]

        # Initialize min scores arrays for each position
        min_scores_lho = np.ones(lho_pard_rho.shape[0])
        min_scores_partner = np.ones(lho_pard_rho.shape[0])
        min_scores_rho = np.ones(lho_pard_rho.shape[0])

        # We will now validate the generated deals against the actual bidding.
        # We take one player at a time, and start with the most difficult one (based on number of bids)
        for player in players:            
            # Call process_bidding for the current player, updating all `min_scores` arrays
            lho_pard_rho, min_scores_lho, min_scores_partner, min_scores_rho = self.process_bidding(
                player, lho_pard_rho, min_scores_lho, min_scores_partner, min_scores_rho, 
                auction, nesw_i, hand, vuln, models, index, actual_bids, n_steps, size, models.bidder_model if player["position"] == 1 else models.opponent_model, self.verbose
            )
            #print(f"Processed bidding for player {player['position']} lho_pard_rho shape", lho_pard_rho.shape)
            if lho_pard_rho.shape[0] == 0:
                sorted_scores = []
                quality = -1
                return lho_pard_rho, sorted_scores, c_hcp, c_shp, quality

        
        min_scores = np.ones(lho_pard_rho.shape[0])

        max_distance = lho_bid_count + 2 * pard_bid_count + rho_bid_count  # Replace with the maximum possible distance in your context
        
        if self.use_distance:
            # Initialize an array to store distances
            distances = np.zeros(lho_pard_rho.shape[0], dtype=np.float16)
            # Calculate the Euclidean distance for each index
            # Small distance is good
            for i in range(lho_pard_rho.shape[0]):
                abs_diff_lho = 1 - min_scores_lho[i]
                abs_diff_partner = 1 - min_scores_partner[i]
                abs_diff_rho = 1 - min_scores_rho[i]
              
                # We have returned a set earlier if no bidding
                #if no_of_bids > 0:
                distances[i] = (abs_diff_lho * lho_bid_count + 2 * abs_diff_partner * pard_bid_count + abs_diff_rho * rho_bid_count)

            #if distances[i] < max_distance * (1-self.bid_accept_threshold_bidding) :
            #    print(abs_diff_lho * lho_bids, 2 * abs_diff_partner * pard_bids, abs_diff_rho * rho_bids, no_of_bids)
            #    print(i, distances[i], hand_to_str(lho_pard_rho[i, 0:1, :], models.n_cards_bidding), round(abs_diff_lho,3), hand_to_str(lho_pard_rho[i, 1:2, :], models.n_cards_bidding),round(abs_diff_partner,3), hand_to_str(lho_pard_rho[i, 2:3, :], models.n_cards_bidding),round(abs_diff_rho,3))

            # Normalize the total distance to a scale between 0 and 100
            if verbose:
                print("Max distance", max_distance, lho_bid_count, pard_bid_count, rho_bid_count)
            scaled_distance_A = ((max_distance - distances) / max_distance)
            #print("scaled_distance_A", scaled_distance_A)

            # Get the indices that would sort min_scores in descending order
            sorted_indices = np.argsort(scaled_distance_A)[::-1]
            sorted_scores = scaled_distance_A[sorted_indices]
        else:
            min_scores = np.minimum(min_scores_rho, min_scores)
            min_scores = np.minimum(min_scores_partner, min_scores)
            min_scores = np.minimum(min_scores_lho, min_scores)
            # Get the indices that would sort min_scores in descending order
            sorted_indices = np.argsort(min_scores)[::-1]
            # Extract scores based on the sorted indices
            sorted_scores = min_scores[sorted_indices]

        #print("sorted_indices",sorted_indices)
        #print("sorted_scores",sorted_scores)

        # Reorder the original lho_pard_rho array based on the sorted indices
        sorted_samples = lho_pard_rho[sorted_indices]

        # We remove the samples with negative scores
        sorted_samples = sorted_samples[sorted_scores >= 0]
        sorted_scores = sorted_scores[sorted_scores >= 0]

        if verbose:
            print("Samples after bidding distance: ", len(sorted_samples), " Threshold: ")
        if len(sorted_scores) == 0:
            return sorted_samples, sorted_scores, c_hcp, c_shp, -1

        # How much to trust the bidding for the samples
        accepted_samples = sorted_samples[sorted_scores >= self.bidding_threshold_sampling]
        if verbose:
            print("Samples after bidding filtering: ", len(accepted_samples), " Threshold: ", self.bidding_threshold_sampling)

        # If we havent found enough samples, just return the minimum number from configuration
        if (len(accepted_samples) < self._min_sample_hands_auction) and extend_samples:
            if self.use_distance:
                if verbose:
                    print(f"Only found {len(sorted_samples[sorted_scores >= self.bid_accept_threshold_bidding])} {self._min_sample_hands_auction}")
            else:
                if verbose:
                    print(f"Only found {len(accepted_samples)} {self._min_sample_hands_auction}")
            accepted_samples = sorted_samples[:self._min_sample_hands_auction]

        if len(accepted_samples) == 0:
            if verbose:
                print(f"No samples found. Extend={extend_samples}")
            return accepted_samples, [], c_hcp, c_shp, -1
        
        if verbose:
            print("Returning", len(accepted_samples), extend_samples)
        sorted_scores = sorted_scores[:len(accepted_samples)]
        quality = np.mean(sorted_scores)

        return accepted_samples, sorted_scores, c_hcp, c_shp, quality

    # shuffle the cards between the 2 hidden hands
    def shuffle_cards_bidding_info(self, n_samples, auction, hand_str, public_hand_str,vuln, known_nesw, h_1_nesw, h_2_nesw, current_trick, hidden_cards, cards_played, shown_out_suits, rng, models):
        hand = binary.parse_hand_f(models.n_cards_bidding)(hand_str)
        if self.verbose:    
            print(f"Called shuffle_cards_bidding_info {n_samples} - {rng.bit_generator.state['state']['state']}")

        # This is a constant and should probably be define globally
        card_hcp = [4, 3, 2, 1, 0, 0, 0, 0] * 4
        cards_in_suit = models.n_cards_play // 4

        # If we are declarer (or dummy) and they did not bid, we will not use bidding info.

        h1_passive = all(item in {"PASS", "PAD_START"} for item in auction[h_1_nesw::4])
        h2_passive = all(item in {"PASS", "PAD_START"} for item in auction[h_2_nesw::4])

        #print(f"h1_passive: {h1_passive}, h2_passive: {h2_passive}")
        use_bidding_info_stats = self.use_biddinginfo and not (h1_passive and h2_passive)

        if use_bidding_info_stats:
            n_steps = binary.calculate_step_bidding_info(auction)

            A = binary.get_auction_binary_sampling(n_steps, auction, known_nesw, hand, vuln, models, models.n_cards_bidding)

            p_hcp, p_shp = models.binfo_model.pred_fun(A)
            if tf.is_tensor(p_hcp):
                p_hcp = p_hcp.numpy()
                p_shp = p_shp.numpy()

            p_hcp = p_hcp.reshape((-1, n_steps, 3))[:, -1, :]
            p_shp = p_shp.reshape((-1, n_steps, 12))[:, -1, :]

            def f_trans_hcp(x): return 4 * x + 10
            def f_trans_shp(x): return 1.75 * x + 3.25

            p_hcp = f_trans_hcp(p_hcp[0, [(h_1_nesw - known_nesw) % 4 - 1, (h_2_nesw - known_nesw) % 4 - 1]])
            p_shp = f_trans_shp(p_shp[0].reshape((3, 4))[[(h_1_nesw - known_nesw) % 4 - 1, (h_2_nesw - known_nesw) % 4 - 1], :])


            # we get an average hcp and distribution from bidding info
            # this average is used to distribute the cards
            # so the final hands are close to stats
            # we need to count the hcp missing and compare it to stats

            missing_hcp = 40 - (binary.get_hcp(hand)[0] + binary.get_hcp(binary.parse_hand_f(models.n_cards_bidding)(public_hand_str))[0])
            if missing_hcp > 0:
                hcp_reduction_factor = round(self.hcp_reduction_factor * np.sum(p_hcp) / missing_hcp, 2)
            else:
                hcp_reduction_factor = 0
                    
            shp_reduction_factor = self.shp_reduction_factor

            if self.verbose:
                print("missing_hcp:", missing_hcp, "hcp_reduction_factor",hcp_reduction_factor, "shp_reduction_factor" ,shp_reduction_factor)

            # acknowledge all played cards
            for i, cards in enumerate(cards_played):
                for c in cards:
                    p_hcp[i] -= card_hcp[c] * hcp_reduction_factor
                    suit = c // cards_in_suit
                    p_shp[i, suit] -= shp_reduction_factor

            if len(current_trick) in {1, 2, 3}:
                indices = [(0, 0), (0, 1), (1, 0), (0, 2)]  # (i, card_index) pairs for each trick length
                for i, card_index in indices[:len(current_trick)]:
                    card = current_trick[card_index]
                    p_hcp[i] -= card_hcp[card] * hcp_reduction_factor
                    p_shp[i, card // cards_in_suit] -= shp_reduction_factor
                    #print("p_hcp: ", p_hcp, "p_shp: ", p_shp)
        
            r_hcp = np.zeros((n_samples, 2)) + p_hcp
            r_shp = np.zeros((n_samples, 2, 4)) + p_shp
        else:
            if self.verbose:
                print(f"We do not use the bidding info. Opponents passive={h1_passive and h2_passive}")
            shp_reduction_factor = 0
            hcp_reduction_factor = 0
            # we do not use the bidding info
            r_hcp = np.ones((n_samples, 2))
            r_shp = np.ones((n_samples, 2, 4))

        n_cards_to_receive = np.array([len(hidden_cards) // 2, len(hidden_cards) - len(hidden_cards) // 2])
        h1_h2 = np.zeros((n_samples, 2, models.n_cards_play), dtype=int)
        cards_received = np.zeros((n_samples, 2), dtype=int)

        # distribute all cards of suits which are known to have shown out
        cards_shownout_suits = []
        for i, suits in enumerate(shown_out_suits):
            for suit in suits:
                for card in filter(lambda x: x // cards_in_suit == suit, hidden_cards):
                    other_hand_i = (i + 1) % 2
                    h1_h2[:, other_hand_i, card] += 1
                    cards_received[:, other_hand_i] += 1
                    if use_bidding_info_stats:
                        p_hcp[other_hand_i] -= card_hcp[card] * hcp_reduction_factor
                        # As we know it is a void, then this will have no effect
                        p_shp[other_hand_i, suit] -= shp_reduction_factor
                    cards_shownout_suits.append(card)

        hidden_cards = [c for c in hidden_cards if c not in cards_shownout_suits]
        ak_cards = [c for c in hidden_cards if c in {0, 1, 8, 9, 16, 17, 24, 25}]
        small_cards = [c for c in hidden_cards if c not in {0, 1, 8, 9, 16, 17, 24, 25}]

        ak_out_i = np.zeros((n_samples, len(ak_cards)), dtype=int)
        # Fill all samples with the missing AK
        ak_out_i[:, :] = np.array(ak_cards)
        # Shuffle AK in all samples
        ak_out_i = np.vectorize(lambda x: rng.permutation(np.copy(x)), signature='(n)->(n)')(ak_out_i)
        small_out_i = np.zeros((n_samples, len(small_cards)), dtype=int)
        small_out_i[:, :] = np.array(small_cards)
        small_out_i = np.vectorize(lambda x: rng.permutation(np.copy(x)), signature='(n)->(n)')(small_out_i)


        s_all = np.arange(n_samples)

        n_max_cards = np.zeros((n_samples, 2), dtype=int) + n_cards_to_receive

        js = np.zeros(n_samples, dtype=int)
        # Distribute AK
        while True:
            s_all_r = s_all[js < ak_out_i.shape[1]]
            if len(s_all_r) == 0:
                break
            #print("r_hcp[s_all_r]: ", r_hcp[s_all_r])

            js_r = js[s_all_r]
            cards = ak_out_i[s_all_r, js_r]
            receivers = distr2_vec(r_shp[s_all_r, :, cards//cards_in_suit], r_hcp[s_all_r], rng)

            can_receive_cards = cards_received[s_all_r, receivers] < n_max_cards[s_all_r, receivers]

            cards_received[s_all_r[can_receive_cards], receivers[can_receive_cards]] += 1
            h1_h2[s_all_r[can_receive_cards], receivers[can_receive_cards], cards[can_receive_cards]] += 1
            # we update stats from bidding_info so lower odds to get next honor card
            # Above we use hcp / 1.2, but here it is fixed to 3
            if use_bidding_info_stats:
                r_hcp[s_all_r[can_receive_cards], receivers[can_receive_cards]] -= 3 * hcp_reduction_factor
                r_shp[s_all_r[can_receive_cards], receivers[can_receive_cards], cards[can_receive_cards] // cards_in_suit] -= shp_reduction_factor
            js[s_all_r[can_receive_cards]] += 1

        js = np.zeros(n_samples, dtype=int)
        while True:
            s_all_r = s_all[js < small_out_i.shape[1]]
            if len(s_all_r) == 0:
                break

            js_r = js[s_all_r]
            cards = small_out_i[s_all_r, js_r]
            receivers = distr_vec(r_shp[s_all_r, :, cards//cards_in_suit], rng)

            can_receive_cards = cards_received[s_all_r, receivers] < n_max_cards[s_all_r, receivers]

            cards_received[s_all_r[can_receive_cards], receivers[can_receive_cards]] += 1
            h1_h2[s_all_r[can_receive_cards], receivers[can_receive_cards], cards[can_receive_cards]] += 1
            r_shp[s_all_r[can_receive_cards], receivers[can_receive_cards], cards[can_receive_cards] // cards_in_suit] -= shp_reduction_factor
            js[s_all_r[can_receive_cards]] += 1

        assert np.sum(h1_h2) == n_samples * np.sum(n_cards_to_receive)

        return h1_h2, use_bidding_info_stats

    def get_opening_lead_scores(self, auction, vuln, models, handsamples, opening_lead_card):
        assert(handsamples.shape[1] == models.n_cards_play)
        cards_in_suit = models.n_cards_play // 4
        #handsamples = handsamples[:1,:]
        contract = bidding.get_contract(auction)

        level = int(contract[0])
        strain = bidding.get_strain_i(contract)
        doubled = int('X' in contract)
        redbld = int('XX' in contract)

        x = np.zeros((handsamples.shape[0], 42), dtype=np.float16)
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
        # Model expect a 32 card deck
        x[:, 10:] = handsamples

        b = np.zeros((handsamples.shape[0], 15), dtype=np.float16)

        n_steps = binary.calculate_step_bidding_info(auction)

        binfo_model = models.binfo_model

        handbidding = handsamples

        # convert the deck if different for play and bidding
        if models.n_cards_play != models.n_cards_bidding:
            handbidding = np.zeros((handsamples.shape[0], models.n_cards_bidding))
            for i in range(handsamples.shape[0]):
                handbidding[i] = binary.parse_hand_f(models.n_cards_bidding)(deck52.handxxto52str(handsamples[i], models.n_cards_play))

        A = binary.get_auction_binary_sampling(n_steps, auction, lead_index, handbidding, vuln, models, models.n_cards_bidding)

        p_hcp, p_shp = binfo_model.pred_fun(A)
        if tf.is_tensor(p_hcp):
            p_hcp = p_hcp.numpy()
            p_shp = p_shp.numpy()

        b[:, :3] = p_hcp.reshape((-1, n_steps, 3))[:, -1, :].reshape((-1, 3))
        b[:, 3:] = p_shp.reshape((-1, n_steps, 12))[:, -1, :].reshape((-1, 12))

        if (contract[1] == "N"):
            lead_softmax = models.lead_nt_model.pred_fun(x, b)
        else:
            lead_softmax = models.lead_suit_model.pred_fun(x, b)
        if self.lead_accept_threshold_suit:
            # We count the probability for each suit
            quartile_probabilities = []
            for row in lead_softmax:
                quartiles = np.array_split(row, 4)  # Split each row into 4 quartiles
                quartile_probs = [np.sum(q) for q in quartiles]  # Sum each quartile
                quartile_probabilities.append(quartile_probs)

            # Convert to NumPy array for easier indexing
            quartile_probabilities = np.array(quartile_probabilities)
            if self.verbose:
                print(quartile_probabilities)
            return quartile_probabilities[:,opening_lead_card // cards_in_suit]
        
        if self.lead_accept_threshold_honors:
            # Reduce each row by splitting it into 4 parts and summing the last 3 elements in each part
            reduced_data = []
            for row in lead_softmax:
                parts = np.array_split(row, 4)  # Split row into 4 parts, each with 8 elements
                reduced_row = []
                for part in parts:
                    reduced_part = np.concatenate([part[:5], [np.sum(part[5:])]])  # Keep first 5 elements, sum the last 3
                    reduced_row.extend(reduced_part)
                reduced_data.append(reduced_row)

            # Convert reduced data to a 2D numpy array for easier handling if needed
            reduced_data = np.array(reduced_data)
            new_lead_index = (opening_lead_card // cards_in_suit) * 6 + min(opening_lead_card % cards_in_suit, 5)
            return reduced_data[:, new_lead_index]

        return lead_softmax[:, opening_lead_card]

    def get_bid_scores(self, nesw_i, partner, auction, vuln, sample_hands, models):
        n_steps = binary.calculate_step_bidding(auction)
        if self.verbose:
            print(f"Get bid scores for samples. First hand {hand_to_str(sample_hands[0])}")

        # convert the deck if different for play and bidding
        if models.n_cards_play != models.n_cards_bidding:
            handbidding = np.zeros((sample_hands.shape[0], models.n_cards_bidding))
            for i in range(sample_hands.shape[0]):
                handbidding[i] = binary.parse_hand_f(models.n_cards_bidding)(deck52.handxxto52str(sample_hands[i], models.n_cards_play))
        else:
            handbidding = sample_hands


        A = binary.get_auction_binary_sampling(n_steps, auction, nesw_i, handbidding, vuln, models, models.n_cards_bidding)

        X = np.zeros((sample_hands.shape[0], n_steps, A.shape[-1]), dtype=np.float16)
        X[:, :, :] = A

        actual_bids, _ = bidding.get_bid_ids(auction, nesw_i, n_steps)
        if partner:
            sample_bids, _ = models.bidder_model.pred_fun_seq(X)
        else:
            sample_bids, _ = models.opponent_model.pred_fun_seq(X)
        if tf.is_tensor(sample_bids):
            sample_bids = sample_bids.numpy()
        sample_bids = sample_bids.reshape((sample_hands.shape[0], n_steps, -1))

        min_scores = np.ones(sample_hands.shape[0])

        # We check the bid for each bidding round
        for i in range(n_steps):
            if actual_bids[i] not in (bidding.BID2ID['PAD_START'], bidding.BID2ID['PAD_END']):
                min_scores = np.minimum(min_scores, sample_bids[:, i, actual_bids[i]])
        return min_scores
    
    def init_rollout_states(self, trick_i, player_i, card_players, player_cards_played, shown_out_suits, current_trick, auction, hand_str, public_hand_str,vuln, models, rng):
        if self.verbose:
            print(f"Called init_rollout_states {self.sample_hands_play} - Contract {bidding.get_contract(auction)} - Player {player_i}")
        rollout_states, bidding_scores, c_hcp, c_shp, quality, probability_of_occurence, lead_scores, play_scores = self.init_rollout_states_iterative(trick_i, player_i, card_players, player_cards_played, shown_out_suits, current_trick, auction, hand_str, public_hand_str,vuln, models, rng)

        return rollout_states, bidding_scores, c_hcp, c_shp, quality, probability_of_occurence, lead_scores, play_scores
    
    def init_rollout_states_iterative(self, trick_i, player_i, card_players, player_cards_played, shown_out_suits, current_trick, auction, hand_str, public_hand_str,vuln, models, rng):
        hand_bidding = binary.parse_hand_f(models.n_cards_bidding)(hand_str)
        n_samples = self.sample_hands_play
        contract = bidding.get_contract(auction)
        #if self.verbose:
        #    print(f"Called init_rollout_states_iterative {n_samples} - Contract {contract} - Player {player_i}")

        leader_i = (player_i - len(current_trick)) % 4
        # Dummy is always 1
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
            hidden_cards_no = len(hidden_cards)
            
            assert hidden_cards_no <= 26, hidden_cards_no

            known_nesw = player_to_nesw_i(player_i, contract)
            h_1_nesw = player_to_nesw_i(hidden_1_i, contract)
            h_2_nesw = player_to_nesw_i(hidden_2_i, contract)
            sample_boards_for_play = self.sample_boards_for_play
            # if hidden_cards_no < 12:
            # sample_boards_for_play = sample_boards_for_play *10
            # if hidden_cards_no < 7:
            #     sample_boards_for_play = sample_boards_for_play // 4
            # The more cards we know the less samples are needed to 
            # With 7 cards left we can generate all possible combinations
            # This should probably be iterative, so we dfon't have to create 5.000 samples for 7 cards
            h1_h2, use_bidding_info_stats = self.shuffle_cards_bidding_info(
                sample_boards_for_play,
                auction,
                hand_str,
                public_hand_str,
                vuln,
                known_nesw,
                h_1_nesw,
                h_2_nesw,
                current_trick,
                hidden_cards,
                [player_cards_played[hidden_1_i], player_cards_played[hidden_2_i]],
                [shown_out_suits[hidden_1_i], shown_out_suits[hidden_2_i]],
                rng,
                models
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
        else:
            # In cheat mode all cards are known
            known_nesw = player_to_nesw_i(player_i, contract)
            states = [np.zeros((1, 13, 298)) for _ in range(4)]
            for k in range(4):
                for i in range(13):
                    states[k][0, i, :32] = card_players[k].x_play[0, i, :32]

        if self.verbose:
            print(f"players_states {states[0].shape[0]} trick {trick_i+1}")

        unique_indices = np.ones(states[0].shape[0]).astype(bool)

        # this is to see how many samples we actually have
        samples = []
        counts = defaultdict(int)  # Dictionary to store counts of each sample  
        for i in range(states[0].shape[0]):
            sample = '%s %s %s %s' % (
                hand_to_str(states[0][i, 0, :32].astype(int)),
                hand_to_str(states[1][i, 0, :32].astype(int)),
                hand_to_str(states[2][i, 0, :32].astype(int)),
                hand_to_str(states[3][i, 0, :32].astype(int)),
            )
            if sample in samples:
                unique_indices[i] = False
            else:
                samples.append(sample)
            
            counts[sample] += 1 

        # Use the unique_indices to filter player_states
        states = [state[unique_indices] for state in states]
        if self.verbose:
            print(f"Unique states {states[0].shape[0]}")

        if use_bidding_info_stats:
            accept, c_hcp, c_shp = self.validate_shape_and_hcp_for_sample(auction, known_nesw, hand_bidding, vuln, h_1_nesw, h_2_nesw, hidden_1_i, hidden_2_i, states, models)
            # If to few examples we ignore the above filtering
            if np.sum(accept) < n_samples:
                accept = np.ones_like(accept).astype(bool)

            states = [state[accept] for state in states]
        else:
            c_hcp, c_shp = None, None
        
        min_bid_scores = np.ones(states[0].shape[0])

        # Loop the samples for each of the 2 hidden hands to check bidding
        # We should generally trust our partners bidding most
        
        for h_i in [hidden_1_i, hidden_2_i]:
            #if (player_i + 2) % 4 == h_i:
            h_i_nesw = player_to_nesw_i(h_i, contract)
            partner = player_i == (h_i + 2) % 4
            bid_scores = self.get_bid_scores(h_i_nesw, partner, auction, vuln, states[h_i][:, 0, :32], models)
            min_bid_scores = np.minimum(min_bid_scores, bid_scores)
            
        # Perhaps this should be calculated more statistical, as we are just taking the bid with the highest score
        # This need to be updated to euclidian distance or logarithmic
        # Round min_bid_scores to 3 decimals
        rounded_scores = np.round(min_bid_scores, 3)

        # Get the indices that would sort the rounded scores in descending order
        sorted_indices = np.argsort(rounded_scores)[::-1]

        sorted_min_bid_scores = min_bid_scores[sorted_indices]
        # print("sorted_min_bid_scores",sorted_min_bid_scores)
        # Sort second dimension within each array in states based on min_bid_scores
        bidding_states = [state[sorted_indices] for state in states]
        #print(bidding_states[0].shape[0])
        valid_bidding_samples = np.sum(sorted_min_bid_scores > self.bid_extend_play_threshold)
        #print(valid_bidding_samples, sample_boards_for_play)
        if bidding_states[0].shape[0] > 200 and valid_bidding_samples > 10:
            # We drop the samples we are not going to use
            bidding_states = [state[sorted_min_bid_scores > self.bid_extend_play_threshold/10] for state in bidding_states]
            sorted_min_bid_scores = sorted_min_bid_scores[sorted_min_bid_scores > self.bid_extend_play_threshold/10]
        assert bidding_states[0].shape[0] > 0, "No samples after checking bidding"

        if self.verbose:
            print(f"Samples {bidding_states[0].shape[0]} before checking opening lead (after shape and hcp and bidding)")

        # We should probably test this after validating the bidding, and not before
        # reject samples inconsistent with the opening lead
        # We will only check opening lead if we have a lot of samples, as we can't trust other will follow the same lead rules
        if self.lead_accept_threshold > 0:
            bidding_states, sorted_min_bid_scores, lead_scores = self.validate_opening_lead_for_sample(trick_i, hidden_1_i, hidden_2_i, current_trick, player_cards_played, models, auction, vuln, bidding_states, sorted_min_bid_scores)
            if self.verbose:
                print(f"Samples {bidding_states[0].shape[0]} after checking lead")
        else:  
            lead_scores = np.ones(bidding_states[0].shape[0])

        assert bidding_states[0].shape[0] > 0, "No samples after opening lead"

        # We should probably test this after validating the bidding, and not before
        if self.play_accept_threshold > 0 and trick_i + 1 <= 11:
            bidding_states, sorted_min_bid_scores, lead_scores, play_scores = self.validate_play_until_now(trick_i, current_trick, leader_i, player_cards_played, hidden_1_i, hidden_2_i, bidding_states, sorted_min_bid_scores, models, contract, lead_scores)
        else:
            play_scores = np.ones(bidding_states[0].shape[0])
        if self.verbose:
            print(f"Samples {bidding_states[0].shape[0]} after checking the play. Trick {trick_i + 1}")

        assert bidding_states[0].shape[0] > 0, "No samples after checking play"

        # Count how many samples we found matching the bidding
        valid_bidding_samples = np.sum(sorted_min_bid_scores > self.bidding_threshold_sampling)
        if self.verbose:
            print("Bidding samples accepted: ",valid_bidding_samples)

        # With only few cards left we will not filter the samples according to the bidding.
        if trick_i + 1 <= 11:
            # trusting the bidding after sampling cards
            # This could probably be set based on number of deals matching or sorted
            if valid_bidding_samples >= self.sample_hands_play: 
                #if self.verbose:
                bidding_states = [state[sorted_min_bid_scores > self.bidding_threshold_sampling] for state in bidding_states]
                lead_scores = lead_scores[sorted_min_bid_scores > self.bidding_threshold_sampling]
                play_scores = play_scores[sorted_min_bid_scores > self.bidding_threshold_sampling]
                sorted_min_bid_scores = sorted_min_bid_scores[sorted_min_bid_scores > self.bidding_threshold_sampling]
                # Randomize the samples, as we have to many
                # Should be based on likeliness of how well the bidding match or opening lead or play or a combination
                # Calculate a combined score for each entry
                combined_scores = sorted_min_bid_scores + lead_scores + play_scores
                # Normalize combined scores to sum to 1 (this forms probabilities for sampling)
                probabilities = combined_scores / np.sum(combined_scores)
                # Perform weighted random permutation
                random_indices = rng.choice(
                    np.arange(bidding_states[0].shape[0]),  # Indices to choose from
                    size=bidding_states[0].shape[0],        # Number of samples
                    replace=False,                    # No replacement (a permutation)
                    p=probabilities                   # Probabilities for weighted randomness
                )
                #random_indices = rng.permutation(bidding_states[0].shape[0])
                bidding_states = [state[random_indices] for state in bidding_states]
                sorted_min_bid_scores = sorted_min_bid_scores[random_indices]
                lead_scores = lead_scores[random_indices]
                play_scores = play_scores[random_indices]
            else:            
                # Count how many samples we found matching the bidding
                valid_bidding_samples = np.sum(sorted_min_bid_scores > self.bid_accept_play_threshold)
                if valid_bidding_samples < self.min_sample_hands_play: 
                    if np.sum(sorted_min_bid_scores > self.bid_extend_play_threshold) == 0:
                        if self.verbose:
                            sys.stderr.write(" We did not find any good samples\n")
                        # We just take top three as we really have no idea about what the bidding means
                        bidding_states = [state[:self.min_sample_hands_play_bad] for state in bidding_states]
                        sorted_min_bid_scores = sorted_min_bid_scores[:self.min_sample_hands_play_bad]
                        lead_scores = lead_scores[:self.min_sample_hands_play_bad]
                        play_scores = play_scores[:self.min_sample_hands_play_bad]
                    else:
                        if self.verbose:
                            sys.stderr.write(" Extending with samples below threshold\n")
                        bidding_states = [state[sorted_min_bid_scores > self.bid_extend_play_threshold] for state in bidding_states]
                        lead_scores = lead_scores[sorted_min_bid_scores > self.bid_extend_play_threshold]
                        play_scores = play_scores[sorted_min_bid_scores > self.bid_extend_play_threshold]
                        sorted_min_bid_scores = sorted_min_bid_scores[sorted_min_bid_scores > self.bid_extend_play_threshold]
                        # Limit to just the minimum needed

                        # Randomize the samples even though they are bad 
                        combined_scores = sorted_min_bid_scores + lead_scores + play_scores
                        # Normalize combined scores to sum to 1 (this forms probabilities for sampling)
                        probabilities = combined_scores / np.sum(combined_scores)
                        # Perform weighted random permutation
                        random_indices = rng.choice(
                            np.arange(bidding_states[0].shape[0]),  # Indices to choose from
                            size=bidding_states[0].shape[0],        # Number of samples
                            replace=False,                    # No replacement (a permutation)
                            p=probabilities                   # Probabilities for weighted randomness
                        )
                        #random_indices = rng.permutation(bidding_states[0].shape[0])
                        bidding_states = [state[random_indices] for state in bidding_states]
                        sorted_min_bid_scores = sorted_min_bid_scores[random_indices]
                        lead_scores = lead_scores[random_indices]
                        play_scores = play_scores[random_indices]

                        # And now select the minimum number required
                        bidding_states = [state[:self.min_sample_hands_play_bad] for state in bidding_states]
                        sorted_min_bid_scores = sorted_min_bid_scores[:self.min_sample_hands_play_bad]
                        lead_scores = lead_scores[:self.min_sample_hands_play_bad]
                        play_scores = play_scores[:self.min_sample_hands_play_bad]
                else:
                    if self.verbose:
                        print("Enough samples above threshold: ",self.bid_accept_play_threshold)
                    bidding_states = [state[sorted_min_bid_scores > self.bid_accept_play_threshold] for state in bidding_states]
                    lead_scores = lead_scores[sorted_min_bid_scores > self.bid_accept_play_threshold]
                    play_scores = play_scores[sorted_min_bid_scores > self.bid_accept_play_threshold]
                    sorted_min_bid_scores = sorted_min_bid_scores[sorted_min_bid_scores > self.bid_accept_play_threshold]
        
        bidding_states = [state[:min(bidding_states[0].shape[0],n_samples)] for state in bidding_states]
        sorted_min_bid_scores = sorted_min_bid_scores[:bidding_states[0].shape[0]]
        lead_scores = lead_scores[:bidding_states[0].shape[0]]
        play_scores = play_scores[:bidding_states[0].shape[0]]
        quality = round(np.mean(sorted_min_bid_scores),5)

        if self.verbose:
            print(f"Returning {min(bidding_states[0].shape[0],n_samples)} {quality:.5f}")
        assert bidding_states[0].shape[0] > 0, "No samples generated for play"

        #print(len(sorted_min_bid_scores), len(lead_scores), bidding_states[0].shape[0])
        #probability_of_occurence = convert_to_probability(sorted_min_bid_scores[:min(bidding_states[0].shape[0],n_samples)])
        probability_of_occurence = convert_to_probability_with_weight(sorted_min_bid_scores, bidding_states, counts)

        return bidding_states, sorted_min_bid_scores,  c_hcp, c_shp, quality, probability_of_occurence, lead_scores, play_scores
    
    def validate_shape_and_hcp_for_sample(self, auction, known_nesw, hand, vuln, h_1_nesw, h_2_nesw, hidden_1_i, hidden_2_i, states, models):
        n_steps = binary.calculate_step_bidding_info(auction)
        cards_in_suit = models.n_cards_play // 4

        A = binary.get_auction_binary_sampling(n_steps, auction, known_nesw, hand, vuln, models, models.n_cards_bidding)

        p_hcp, p_shp = models.binfo_model.pred_fun(A)

        if tf.is_tensor(p_hcp):
            p_hcp = p_hcp.numpy()
            p_shp = p_shp.numpy()

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
            print(f"c_hcp:{c_hcp[0]:0.2f}")
            print(f"c_shp:{c_shp[0]}")

        accept_hcp = np.ones(states[0].shape[0]).astype(bool)

        for i in range(2):
            if np.round(c_hcp[i]) >= 11:
                accept_hcp &= binary.get_hcp(states[[hidden_1_i, hidden_2_i][i]][:, 0, :32]) >= np.round(c_hcp[i]) - 5

        accept_shp = np.ones(states[0].shape[0]).astype(bool)

        for i in range(2):
            for j in range(4):
                if np.round(c_shp[i, j] < 1.5):
                    accept_shp &= np.sum(states[[hidden_1_i, hidden_2_i][i]][:, 0, (j*cards_in_suit):((j+1)*cards_in_suit)], axis=1) <= np.round(c_shp[i, j]) + 2
                if np.round(c_shp[i, j] >= 5):
                    accept_shp &= np.sum(states[[hidden_1_i, hidden_2_i][i]][:, 0, (j*cards_in_suit):((j+1)*cards_in_suit)], axis=1) >= np.round(c_shp[i, j]) - 1
                if np.round(c_shp[i, j] >= 6):
                    accept_shp &= np.sum(states[[hidden_1_i, hidden_2_i][i]][:, 0, (j*cards_in_suit):((j+1)*cards_in_suit)], axis=1) >= np.round(c_shp[i, j])

        accept = accept_hcp & accept_shp

        return accept, c_hcp, c_shp.flatten()

    def validate_opening_lead_for_sample(self, trick_i, hidden_1_i, hidden_2_i, current_trick, player_cards_played, models, auction, vuln, states, bid_scores):
        if self.verbose:
            print("validate_opening_lead_for_sample:", "lead_accept_threshold:", self.lead_accept_threshold, 
                  "lead_accept_threshold_partner_trust:", self.lead_accept_threshold_partner_trust,
                  "min_sample_hands_play:", self.min_sample_hands_play, "lead_accept_threshold_suit:", self.lead_accept_threshold_suit, 
                  "lead_accept_threshold_honors:", self.lead_accept_threshold_honors)
        # Only make the test if opening leader (0) is hidden
        # The primary idea is to filter away hands, that lead the Q as it denies the K
        lead_scores = np.zeros(states[0].shape[0])
        # Opening leader is in first seat (0)
        if (hidden_1_i == 0 or hidden_2_i == 0): 
            # Here we should probably remove leads that should not be possible.
            # But impossible lead should get a very low score from the neural network, so perhaps it is overkill
            if states[0].shape[0] >= self.min_sample_hands_play:
                if (hidden_2_i == 3):
                    # We are RHO and trust partners lead
                    lead_accept_threshold = self.lead_accept_threshold + self.lead_accept_threshold_partner_trust
                    if self.verbose:
                        print(f"RHO and trust partners lead: {lead_accept_threshold:0.3f}")
                else: 
                    # How much trust that opponents would have lead the actual card from the hand sampled
                    lead_accept_threshold = self.lead_accept_threshold

                opening_lead = current_trick[0] if trick_i == 0 else player_cards_played[0][0]
                lead_scores = self.get_opening_lead_scores(auction, vuln, models, states[0][:, 0, :models.n_cards_play], opening_lead)
                if tf.is_tensor(lead_scores):
                    lead_scores = lead_scores.numpy()

                while np.sum(lead_scores >= lead_accept_threshold) < self.min_sample_hands_play and lead_accept_threshold > 0:
                    # We are RHO and trust partners lead
                    lead_accept_threshold *= 0.5
                    if lead_accept_threshold < 0.001:
                        if self.verbose:
                            print("Skipping validation of opening lead as no samples above threshold")           
                        sys.stderr.write("Skipping validation of opening lead as no samples above threshold\n") 
                        sys.stderr.write(f"lead_scores: {lead_scores}\n") 
                        return states, bid_scores, lead_scores

                # If we did not find 2 samples we ignore the test for opening lead
                if np.sum(lead_scores >= lead_accept_threshold) > 1:
                    states = [state[lead_scores > lead_accept_threshold] for state in states]
                    bid_scores = bid_scores[lead_scores > lead_accept_threshold]
                    lead_scores = lead_scores[lead_scores > lead_accept_threshold]
                else:
                    if self.verbose:
                        print("Skipping validation of opening as only 1 or less samples")            

            else:
                if self.verbose:
                    print(f"Skipping validation of opening lead due to to few samples: {states[0].shape[0]} need {self.min_sample_hands_play}")            
        else:
            if self.verbose:
                print("Opening lead not from hidden hand")            
        return states, bid_scores, lead_scores

    # Check that the play until now is expected with the samples
    # In principle we do this to eliminated hands, where the card played is inconsistent with the sample
    # We should probably only validate partner as he follow our rules (what is in the neural net)
    # But it will also help us in eleminating hands, where the play would be illogical
    def validate_play_until_now(self, trick_i, current_trick, leader_i, player_cards_played, hidden_1_i, hidden_2_i, states, bidding_scores, models, contract, lead_scores):
        # If they don't cover they dont have that card
        # Should be implemented as a logical rule TODO
        if self.verbose:
            print("Validating play")
            print(trick_i, current_trick, leader_i, player_cards_played, hidden_1_i, hidden_2_i, states[0].shape[0])
        min_play_scores = np.ones(states[0].shape[0])
        strain_i = bidding.get_strain_i(contract)
        # Select playing models based on NT orsuit
        playermodelindex = 0 if strain_i == 0 else 4

        # Trust in the play until now
        play_accept_threshold = self.play_accept_threshold

        for p_i in [hidden_1_i, hidden_2_i]:

            if trick_i == 0 and p_i == 0:
                continue
            # We will not test declarer
            if p_i == 3:
                # We are defending, so we trust our partner
                play_accept_threshold = self.play_accept_threshold_partner 
                continue
            card_played_current_trick = []
            for i, card in enumerate(current_trick):
                if (leader_i + i) % 4 == p_i:
                    card_played_current_trick.append(card)

            # If the opening lead isn't included in the neural network for Lefty, then send from trick 2
            if p_i == 0 and not models.opening_lead_included:
                cards_played_prev_tricks = player_cards_played[p_i][1:trick_i]
            else:
                cards_played_prev_tricks = player_cards_played[p_i][:trick_i]

            cards_played = cards_played_prev_tricks + card_played_current_trick

            # No cards played except opening lead, so just return
            if len(cards_played) == 0:
                return states, bidding_scores, lead_scores, min_play_scores
            
            if self.verbose:
                print(f"cards_played by {p_i} {cards_played}")
            # We have 11 rounds of play in the neural network, but might have only 10 for Lefty
            if p_i == 0 and not models.opening_lead_included:
                n_tricks_pred = trick_i + len(card_played_current_trick) - 1
            else:
                n_tricks_pred = trick_i + len(card_played_current_trick)
                
            # Depending on suit or NT we must select the right model
            # 0-3 is for NT 4-7 is for suit
            # When the player is instantiated the right model is selected, but here we get it from the configuration
            # print("trick_i", trick_i, "len(card_played_current_trick)", len(card_played_current_trick), "n_tricks_pred", n_tricks_pred)
            p_cards = models.player_models[p_i+playermodelindex].pred_fun(states[p_i][:, :n_tricks_pred, :])
            if tf.is_tensor(p_cards):
                p_cards = p_cards.numpy()
            card_scores = p_cards[:, np.arange(len(cards_played)), cards_played]
            #print(card_scores.shape)
            #x = states[p_i][:, :n_tricks_pred, :]
            #for i in range(max(10,x.shape[0])):
            #    print(hand_to_str(x[i,0,0:32],models.n_cards_play), card_scores[i, :])
            # The opening lead is validated elsewhere, so we just change the score to 1 for all samples
            if p_i == 0 and models.opening_lead_included:
                card_scores[:, 0] = 1
            #print(f"card_scores {card_scores.flatten()}")

            min_play_scores = np.minimum(min_play_scores, np.min(card_scores, axis=1))
            
        #print(f"card_scores combined {min_play_scores}")

        if self.verbose:
            print(f"Found deals above play threshold: {np.sum(min_play_scores > play_accept_threshold)} play_accept_threshold={play_accept_threshold}")

        while np.sum(min_play_scores > play_accept_threshold) < self.min_play_accept_threshold_samples and play_accept_threshold > 0:
            play_accept_threshold -= 0.01
            #print(f"play_accept_threshold {play_accept_threshold:0.3f} reduced")
        
        s_accepted = min_play_scores > play_accept_threshold

        states = [state[s_accepted] for state in states]
        lead_scores = lead_scores[s_accepted]
        bidding_scores = bidding_scores[s_accepted]
        min_play_scores = min_play_scores[s_accepted]
        return states, bidding_scores, lead_scores, min_play_scores
    