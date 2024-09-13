import sys
import time

import numpy as np

import binary

from bidding import bidding
import deck52
from util import get_all_hidden_cards, calculate_seed, convert_to_probability
from configparser import ConfigParser
from util import hand_to_str

np.set_printoptions(precision=2, suppress=True, linewidth=220, threshold=np.inf)

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

    def __init__(self, lead_accept_threshold, lead_accept_threshold_partner_trust, bidding_threshold_sampling, play_accept_threshold, min_play_accept_threshold_samples, bid_accept_play_threshold, bid_accept_threshold_bidding, bid_extend_play_threshold, sample_hands_auction, min_sample_hands_auction, sample_boards_for_auction, sample_boards_for_auction_opening_lead, sample_hands_opening_lead, sample_hands_play, min_sample_hands_play, min_sample_hands_play_bad, sample_boards_for_play, use_biddinginfo, use_distance, no_samples_when_no_search, exclude_samples, no_biddingqualitycheck_after_bid_count, hcp_reduction_factor, shp_reduction_factor,verbose):
        self.lead_accept_threshold_partner_trust = lead_accept_threshold_partner_trust
        self.lead_accept_threshold = lead_accept_threshold
        self.bidding_threshold_sampling = bidding_threshold_sampling
        self.play_accept_threshold = play_accept_threshold
        self.min_play_accept_threshold_samples = min_play_accept_threshold_samples
        self.bid_accept_play_threshold = bid_accept_play_threshold
        self.bid_accept_threshold_bidding = bid_accept_threshold_bidding
        self.bid_extend_play_threshold = bid_extend_play_threshold
        self._sample_hands_auction = sample_hands_auction
        self._min_sample_hands_auction = min_sample_hands_auction
        self.sample_boards_for_auction = sample_boards_for_auction
        self.sample_boards_for_auction_opening_lead = sample_boards_for_auction_opening_lead
        self.sample_hands_opening_lead = sample_hands_opening_lead
        self.sample_hands_play = sample_hands_play
        self.min_sample_hands_play = min_sample_hands_play
        self.min_sample_hands_play_bad = min_sample_hands_play_bad
        self.sample_boards_for_play = sample_boards_for_play
        self.use_bidding_info = use_biddinginfo
        self.use_distance = use_distance
        self.no_samples_when_no_search = no_samples_when_no_search
        self.exclude_samples = exclude_samples
        self.no_biddingqualitycheck_after_bid_count = no_biddingqualitycheck_after_bid_count
        self.hcp_reduction_factor = hcp_reduction_factor
        self.shp_reduction_factor = shp_reduction_factor
        self.verbose = verbose


    @classmethod
    def from_conf(cls, conf: ConfigParser, verbose=False) -> "Sample":
        lead_accept_threshold = float(conf['sampling']['lead_accept_threshold'])
        lead_accept_threshold_partner_trust = float(conf['sampling']['lead_accept_threshold_partner_trust'])
        bidding_threshold_sampling = float(conf['sampling']['bidding_threshold_sampling'])
        play_accept_threshold = float(conf['sampling']['play_accept_threshold'])
        min_play_accept_threshold_samples = conf.getint('sampling','min_play_accept_threshold_samples',fallback=20)
        bid_accept_play_threshold = float(conf['sampling']['bid_accept_play_threshold'])
        bid_accept_threshold_bidding = float(conf['sampling']['bid_accept_threshold_bidding'])
        bid_extend_play_threshold = float(conf['sampling'].get('bid_extend_play_threshold', 0))
        exclude_samples = float(conf['sampling'].get('exclude_samples', 0))
        sample_hands_auction = int(conf['sampling']['sample_hands_auction'])
        min_sample_hands_auction = int(conf['sampling']['min_sample_hands_auction'])
        sample_boards_for_auction = int(conf['sampling']['sample_boards_for_auction'])

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
        return cls(lead_accept_threshold, lead_accept_threshold_partner_trust, bidding_threshold_sampling, play_accept_threshold, min_play_accept_threshold_samples, bid_accept_play_threshold, bid_accept_threshold_bidding, bid_extend_play_threshold, sample_hands_auction, min_sample_hands_auction, sample_boards_for_auction, sample_boards_for_auction_opening_lead, sample_hands_opening_lead, sample_hands_play, min_sample_hands_play, min_sample_hands_play_bad, sample_boards_for_play, use_biddinginfo, use_distance, no_samples_when_no_search, exclude_samples, no_biddingquality_after_bid_count, hcp_reduction_factor, shp_reduction_factor, verbose)

    @property
    def sample_hands_auction(self):
        return self._sample_hands_auction

    @property
    def min_sample_hands_auction(self):
        return self._min_sample_hands_auction

    def sample_cards_vec(self, n_samples, c_hcp, c_shp, my_hand, rng, n_cards=32):
        if self.verbose:
            print("sample_cards_vec generating", n_samples)
            t_start = time.time()
        deck = np.ones(n_cards, dtype=int)
        cards_in_suit = n_cards // 4

        if self.verbose:
            print("sample_cards_vec cards_in_suit", cards_in_suit)
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

        #c_hcp = (lambda x: 4 * x + 10)(p_hcp.copy())
        c_shp = c_shp.reshape((3, 4))

        r_hcp = np.zeros((n_samples, 3)) + c_hcp
        r_shp = np.zeros((n_samples, 3, 4)) + c_shp

        lho_pard_rho = np.zeros((n_samples, 3, n_cards), dtype=int)
        cards_received = np.zeros((n_samples, 3), dtype=int)

        # calculate missing hcp
        missing_hcp = 40 - binary.get_hcp(np.array([my_hand]))[0]


        if missing_hcp > 0:
            hcp_reduction_factor = self.hcp_reduction_factor * np.sum(r_hcp[0]) / missing_hcp
        else:
            hcp_reduction_factor = 0

        if self.verbose:
            print("Missing HCP:", missing_hcp)
            print("Expected HCP:",r_hcp[0])
            print("hcp_reduction_factor:", hcp_reduction_factor, self.hcp_reduction_factor)            

        # print(ak_out_i)
        # all AK's in the same hand
        if (ak_out_i.shape[1] != 0):
            # print(ak_out_i.shape[1])
            ak_out_i = np.vectorize(rng.permutation, signature='(n)->(n)')(ak_out_i)
        small_out_i = np.vectorize(rng.permutation, signature='(n)->(n)')(small_out_i)

        s_all = np.arange(n_samples)

        # distribute A and K
        js = np.zeros(n_samples, dtype=int)
        while np.min(js) < ak_out_i.shape[1]:
            cards = ak_out_i[s_all, js]
            receivers = distr2_vec(r_shp[s_all, :, cards//cards_in_suit], r_hcp, rng)

            can_receive_cards = cards_received[s_all, receivers] < 13

            cards_received[s_all[can_receive_cards], receivers[can_receive_cards]] += 1
            lho_pard_rho[s_all[can_receive_cards], receivers[can_receive_cards], cards[can_receive_cards]] += 1
            r_hcp[s_all[can_receive_cards], receivers[can_receive_cards]] -= 3 * hcp_reduction_factor
            r_shp[s_all[can_receive_cards], receivers[can_receive_cards], cards[can_receive_cards] // cards_in_suit] -= self.shp_reduction_factor
            js[can_receive_cards] += 1

        # distribute small cards
        js = np.zeros(n_samples, dtype=int)
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
            #loop_counter += 1  # Increment the loop counter
            #if loop_counter >= 250:  # Check if the counter reaches 76
            #    print("Loop counter >= 76")
            #    break  #


        # re-apply constraints
        # This is in principle just to reduce the number of samples for performance
        accept_hcp = np.ones(n_samples).astype(bool)
        
        for i in range(3):
            if np.round(c_hcp[i]) >= 11:
                accept_hcp &= binary.get_hcp(lho_pard_rho[:, i, :]) >= np.round(c_hcp[i]) - 5

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

        accepted = np.sum(accept)
        if self.verbose:
            print(f'sample_cards_vec took {(time.time() - t_start):0.4f} Deals: {accepted}')

        # If we have filtered to many away just return all samples - performance            
        if accepted >= n_samples / 2:
            return lho_pard_rho[accept]
        else:
            return lho_pard_rho

    def get_bidding_info(self, n_steps, auction, nesw_i, hand, vuln, models):
        assert n_steps > 0, "n_steps should be greater than zero"
        A = binary.get_auction_binary_sampling(n_steps, auction, nesw_i, hand, vuln, models, models.n_cards_bidding)
        p_hcp, p_shp = models.binfo_model.model(A)

        p_hcp = p_hcp.reshape((-1, n_steps, 3))[:, -1, :]
        p_shp = p_shp.reshape((-1, n_steps, 12))[:, -1, :]

        c_hcp = (lambda x: 4 * x + 10)(p_hcp.copy())
        c_shp = (lambda x: 1.75 * x + 3.25)(p_shp.copy())

        if self.verbose:
            print("Player: ", 'NESW'[nesw_i], "Hand: ", hand_to_str(hand[0],models.n_cards_bidding))
            print("HCP: ", c_hcp)
            print("Shape: ", c_shp)

        return c_hcp, c_shp

        
    def sample_cards_auction(self, auction, nesw_i, hand_str, vuln, n_samples, rng, models):
        hand = binary.parse_hand_f(models.n_cards_bidding)(hand_str)
        n_steps = binary.calculate_step_bidding_info(auction)
        bids = 4 if models.model_version >= 2 else 3
        if self.verbose:
            print("sample_cards_auction, nsteps=", n_steps)
            print("NS: ", models.ns, "EW: ", models.ew, "Auction: ", auction)
            print("hand", hand_str)
            print("nesw_i", nesw_i)
            print("n_samples", n_samples)

        c_hcp, c_shp = self.get_bidding_info(n_steps, auction, nesw_i, hand, vuln, models)        
          
        lho_pard_rho = self.sample_cards_vec(n_samples, c_hcp[0], c_shp[0], hand.reshape(models.n_cards_bidding), rng, models.n_cards_bidding)

        # Consider saving the generated boards, and add the result from previous sampling to this output
        n_samples = lho_pard_rho.shape[0]
        if self.verbose:
            print(f"n_samples {n_samples} from bidding info")

        n_steps = binary.calculate_step_bidding(auction)
        if self.verbose:
            print("n_steps", n_steps)

        # The hand used as input is our hand, but it will be overwritten with the sampled hand for that player
        A_lho = binary.get_auction_binary_sampling(n_steps, auction, (nesw_i + 1) % 4, hand, vuln, models, models.n_cards_bidding)
        A_pard = binary.get_auction_binary_sampling(n_steps, auction, (nesw_i + 2) % 4, hand, vuln, models, models.n_cards_bidding)
        A_rho = binary.get_auction_binary_sampling(n_steps, auction, (nesw_i + 3) % 4, hand, vuln, models, models.n_cards_bidding)
        #print("RHO: ", n_steps, auction, (nesw_i + 3) % 4, hand, vuln, models)

        if models.model_version == 0 or models.ns == -1 :
            index = 0
        else:
            index = 2

        size = 7 + models.n_cards_bidding + index + bids*40

        X_lho = np.zeros((n_samples, n_steps, size), dtype=np.float16)
        X_pard = np.zeros((n_samples, n_steps, size), dtype=np.float16)
        X_rho = np.zeros((n_samples, n_steps, size), dtype=np.float16)

        X_lho[:, :, :] = A_lho
        X_lho[:, :, 7+index:7+models.n_cards_bidding+index] = lho_pard_rho[:, 0:1, :]
        X_lho[:, :, 2+index] = (binary.get_hcp(lho_pard_rho[:, 0, :]).reshape((-1, 1)) - 10) / 4
        X_lho[:, :, 3+index:7+index] = (binary.get_shape(lho_pard_rho[:, 0, :]).reshape((-1, 1, 4)) - 3.25) / 1.75
        if self.verbose:
            print("get_bid_ids", n_steps, auction, (nesw_i + 2) % 4)
        lho_actual_bids = bidding.get_bid_ids(auction, (nesw_i + 1) % 4, n_steps)
        lho_sample_bids = models.opponent_model.model_seq(X_lho)[0].reshape((n_samples, n_steps, -1))
        if self.verbose:
            print("Fetched LHO bidding")

        X_pard[:, :, :] = A_pard
        X_pard[:, :, 7+index:7+models.n_cards_bidding+index] = lho_pard_rho[:, 1:2, :]
        X_pard[:, :, 2+index] = (binary.get_hcp(lho_pard_rho[:, 1, :]).reshape((-1, 1)) - 10) / 4
        X_pard[:, :, 3+index:7+index] = (binary.get_shape(lho_pard_rho[:, 1, :]).reshape((-1, 1, 4)) - 3.25) / 1.75
        if self.verbose:
            print("get_bid_ids", n_steps, auction, (nesw_i + 2) % 4)
        pard_actual_bids = bidding.get_bid_ids(auction, (nesw_i + 2) % 4, n_steps)
        pard_sample_bids = models.bidder_model.model_seq(X_pard)[0].reshape((n_samples, n_steps, -1))
        if self.verbose:
            print("Fetched partner bidding")

        X_rho[:, :, :] = A_rho
        X_rho[:, :, 7+index:7+models.n_cards_bidding+index] = lho_pard_rho[:, 2:, :]
        X_rho[:, :, 2+index] = (binary.get_hcp(lho_pard_rho[:, 2, :]).reshape((-1, 1)) - 10) / 4
        X_rho[:, :, 3+index:7+index] = (binary.get_shape(lho_pard_rho[:, 2, :]).reshape((-1, 1, 4)) - 3.25) / 1.75
        if self.verbose:
            print("get_bid_ids", n_steps, auction, (nesw_i + 2) % 4)
        rho_actual_bids = bidding.get_bid_ids(auction, (nesw_i + 3) % 4, n_steps)
        rho_sample_bids = models.opponent_model.model_seq(X_rho)[0].reshape((n_samples, n_steps, -1))
        if self.verbose:
            print("Fetched RHO bidding")

        # Consider having scores for partner and opponents
        # Current implementation should be updated due to long sequences is difficult to match
        
        min_scores = np.ones(n_samples)
        min_scores_lho = np.ones(n_samples)
        min_scores_partner = np.ones(n_samples)
        min_scores_rho = np.ones(n_samples)

        lho_bids = 0
        pard_bids = 0
        rho_bids = 0

        for i in range(n_steps):
            if lho_actual_bids[i] not in (bidding.BID2ID['PAD_START'], bidding.BID2ID['PAD_END']):
                min_scores_lho = np.minimum(min_scores_lho, lho_sample_bids[:, i, lho_actual_bids[i]])
                lho_bids += 1
                #print(lho_actual_bids[i])
                # if (lho_actual_bids[i] == 31):
                #     for j in range(n_samples):
                #         if (lho_sample_bids[j, i, lho_actual_bids[i]] >0.3):
                #             print(hand_to_str(lho_pard_rho[j, 0:1, :]))
                #             print(lho_sample_bids[j, i, lho_actual_bids[i]])
                
            if pard_actual_bids[i] not in (bidding.BID2ID['PAD_START'], bidding.BID2ID['PAD_END']):
                min_scores_partner = np.minimum(min_scores_partner, pard_sample_bids[:, i, pard_actual_bids[i]])
                pard_bids += 1
            if rho_actual_bids[i] not in (bidding.BID2ID['PAD_START'], bidding.BID2ID['PAD_END']):
                min_scores_rho = np.minimum(min_scores_rho, rho_sample_bids[:, i, rho_actual_bids[i]])
                rho_bids += 1

        if self.verbose:
            print("lho_bids", lho_bids, "pard_bids", pard_bids, "rho_bids", rho_bids)

        #print(min_scores_rho)
        no_of_bids = lho_bids + pard_bids + rho_bids
        
        if self.use_distance:
            # Initialize an array to store distances
            distances = np.zeros(n_samples, dtype=np.float16)
            # Calculate the Euclidean distance for each index
            # Small distance is good
            for i in range(n_samples):
                abs_diff_lho = np.abs(min_scores[i] - min_scores_lho[i])
                #print(hand_to_str(lho_pard_rho[i, 0:1, :], models.n_cards_bidding), abs_diff_lho)
                abs_diff_partner = np.abs(min_scores[i] - min_scores_partner[i])
                #print(hand_to_str(lho_pard_rho[i, 1:2, :], models.n_cards_bidding), abs_diff_partner)
                abs_diff_rho = np.abs(min_scores[i] - min_scores_rho[i])
                #print(hand_to_str(lho_pard_rho[i, 2:3, :], models.n_cards_bidding), abs_diff_rho)
                
                if no_of_bids > 0:
                    distances[i] = (abs_diff_lho * lho_bids + 2 * abs_diff_partner * pard_bids + abs_diff_rho * rho_bids) / no_of_bids
                # Increase the distance if any absolute score is less than 0.01 (exclude samples) - in principle discarding that sample
                if abs_diff_partner > 1 - self.exclude_samples: 
                    # we do not want to exclude any samples for the oppponents
                    #or abs_diff_partner < self.exclude_samples or abs_diff_rho < self.exclude_samples:
                    distances[i] += 10
                #if min_scores_rho[i] >= 0.99:
                #    print(hand_to_str(lho_pard_rho[i, 0:1, :], models.n_cards_bidding), round(abs_diff_lho,3), hand_to_str(lho_pard_rho[i, 1:2, :], models.n_cards_bidding),round(abs_diff_partner,3), hand_to_str(lho_pard_rho[i, 2:3, :], models.n_cards_bidding),round(abs_diff_rho,3)
                  
            if no_of_bids > 0:
                # Normalize the total distance to a scale between 0 and 100
                max_distance = lho_bids + 2 * pard_bids + rho_bids  # Replace with the maximum possible distance in your context
                if self.verbose:
                    print("Max distance", max_distance)
                scaled_distance_A = ((max_distance - distances) / max_distance)

                # Get the indices that would sort min_scores in descending order
                sorted_indices = np.argsort(scaled_distance_A)[::-1]
                sorted_scores = scaled_distance_A[sorted_indices]
            else:
                sorted_indices = np.argsort(min_scores)[::-1]
                sorted_scores = min_scores[sorted_indices]
        else:
            min_scores = np.minimum(min_scores_rho, min_scores)
            min_scores = np.minimum(min_scores_partner, min_scores)
            min_scores = np.minimum(min_scores_lho, min_scores)
            # Get the indices that would sort min_scores in descending order
            sorted_indices = np.argsort(min_scores)[::-1]
            # Extract scores based on the sorted indices
            sorted_scores = min_scores[sorted_indices]

        # Reorder the original lho_pard_rho array based on the sorted indices
        sorted_samples = lho_pard_rho[sorted_indices]

        # How much to trust the bidding for the samples
        accepted_samples = sorted_samples[sorted_scores >= self.bidding_threshold_sampling]
        if self.verbose:
            print("Samples after bidding filtering: ", len(accepted_samples), " Threshold: ", self.bidding_threshold_sampling)
        good_quality = True

        # If we havent found enough samples, just return the minimum number from configuration
        # It could be an idea to add an extra sampling in a later version
        if len(accepted_samples) < self._min_sample_hands_auction:
            if self.use_distance:
                good_quality = len(sorted_samples[sorted_scores >= self.bid_accept_threshold_bidding]) > 2 or n_steps*4 > self.no_biddingqualitycheck_after_bid_count
                if self.verbose:
                    print(f"Only found {len(sorted_samples[sorted_scores >= self.bid_accept_threshold_bidding])} {self._min_sample_hands_auction}")
            else:
                if self.verbose:
                    print(f"Only found {len(accepted_samples)} {self._min_sample_hands_auction}")
                # If we found 2 boards with OK bidding we accept the quality
                good_quality = len(accepted_samples) > 2 or n_steps*4 > self.no_biddingqualitycheck_after_bid_count
            accepted_samples = sorted_samples[:self._min_sample_hands_auction]

        sorted_scores = sorted_scores[:len(accepted_samples)]
        return accepted_samples, sorted_scores, c_hcp[0], c_shp[0], good_quality

    # shuffle the cards between the 2 hidden hands
    def shuffle_cards_bidding_info(self, n_samples, auction, hand_str, public_hand_str,vuln, known_nesw, h_1_nesw, h_2_nesw, current_trick, hidden_cards, cards_played, shown_out_suits, rng, models):
        hand = binary.parse_hand_f(models.n_cards_bidding)(hand_str)
        missing_hcp = 40 - (binary.get_hcp(hand)[0] + binary.get_hcp(binary.parse_hand_f(models.n_cards_bidding)(public_hand_str))[0])

        if self.verbose:
            print("missing_hcp:", missing_hcp)

        n_cards_to_receive = np.array([len(hidden_cards) // 2, len(hidden_cards) - len(hidden_cards) // 2])

        #if self.verbose:
        #print("shuffle_cards_bidding_info cards to sample: ", n_cards_to_receive)
        n_steps = binary.calculate_step_bidding_info(auction)

        A = binary.get_auction_binary_sampling(n_steps, auction, known_nesw, hand, vuln, models, models.n_cards_bidding)

        p_hcp, p_shp = models.binfo_model.model(A)

        p_hcp = p_hcp.reshape((-1, n_steps, 3))[:, -1, :]
        p_shp = p_shp.reshape((-1, n_steps, 12))[:, -1, :]

        def f_trans_hcp(x): return 4 * x + 10
        def f_trans_shp(x): return 1.75 * x + 3.25

        #print("p_hcp: ", p_hcp, "p_shp: ", p_shp)

        p_hcp = f_trans_hcp(p_hcp[0, [(h_1_nesw - known_nesw) % 4 - 1, (h_2_nesw - known_nesw) % 4 - 1]])
        p_shp = f_trans_shp(p_shp[0].reshape((3, 4))[[(h_1_nesw - known_nesw) % 4 - 1, (h_2_nesw - known_nesw) % 4 - 1], :])

        #print("p_hcp: ", p_hcp, "p_shp: ", p_shp)

        h1_h2 = np.zeros((n_samples, 2, 32), dtype=int)
        cards_received = np.zeros((n_samples, 2), dtype=int)

        card_hcp = [4, 3, 2, 1, 0, 0, 0, 0] * 4

        # we get an average hcp and distribution from bidding info
        # this average is used to distribute the cards
        # so the final hands are close to stats
        # we need to count the hcp missing and compare it to stats

        if missing_hcp > 0:
            hcp_reduction_factor = self.hcp_reduction_factor * np.sum(p_hcp) / missing_hcp
        else:
            hcp_reduction_factor = 0
            
        shp_reduction_factor = self.shp_reduction_factor

        if self.verbose:
            print("hcp_reduction_factor",hcp_reduction_factor, "shp_reduction_factor" ,shp_reduction_factor)

        # acknowledge all played cards
        for i, cards in enumerate(cards_played):
            for c in cards:
                p_hcp[i] -= card_hcp[c] * hcp_reduction_factor
                suit = c // 8
                p_shp[i, suit] -= shp_reduction_factor

        if len(current_trick) == 1:
            # RHO played a card
            i = 0
            p_hcp[i] -= card_hcp[current_trick[0]] * hcp_reduction_factor
            suit = current_trick[0] // 8
            p_shp[i, suit] -= shp_reduction_factor
        if len(current_trick) == 2:
            # RHO played second card
            i = 0
            p_hcp[i] -= card_hcp[current_trick[1]] * hcp_reduction_factor
            suit = current_trick[1] // 8
            p_shp[i, suit] -= shp_reduction_factor
        if len(current_trick) == 3:
            # LHO played first card
            i = 1
            p_hcp[i] -= card_hcp[current_trick[0]] * hcp_reduction_factor
            suit = current_trick[0] // 8
            p_shp[i, suit] -= shp_reduction_factor
            # RHO played third card
            i = 0
            p_hcp[i] -= card_hcp[current_trick[2]] * hcp_reduction_factor
            suit = current_trick[2] // 8
            p_shp[i, suit] -= shp_reduction_factor

        #print("p_hcp: ", p_hcp, "p_shp: ", p_shp)
    
        # distribute all cards of suits which are known to have shown out
        cards_shownout_suits = []
        for i, suits in enumerate(shown_out_suits):
            for suit in suits:
                for card in filter(lambda x: x // 8 == suit, hidden_cards):
                    other_hand_i = (i + 1) % 2
                    h1_h2[:, other_hand_i, card] += 1
                    cards_received[:, other_hand_i] += 1
                    p_hcp[other_hand_i] -= card_hcp[card] * hcp_reduction_factor
                    # As we know it is a void, then this will have no effect
                    p_shp[other_hand_i, suit] -= shp_reduction_factor
                    cards_shownout_suits.append(card)

        hidden_cards = [c for c in hidden_cards if c not in cards_shownout_suits]
        ak_cards = [c for c in hidden_cards if c in {0, 1, 8, 9, 16, 17, 24, 25}]
        small_cards = [c for c in hidden_cards if c not in {0, 1, 8, 9, 16, 17, 24, 25}]
        
        ak_out_i = np.zeros((n_samples, len(ak_cards)), dtype=int)
        ak_out_i[:, :] = np.array(ak_cards)
        ak_out_i = np.vectorize(lambda x: rng.permutation(np.copy(x)), signature='(n)->(n)')(ak_out_i)
        small_out_i = np.zeros((n_samples, len(small_cards)), dtype=int)
        small_out_i[:, :] = np.array(small_cards)
        small_out_i = np.vectorize(lambda x: rng.permutation(np.copy(x)), signature='(n)->(n)')(small_out_i)

        r_hcp = np.zeros((n_samples, 2)) + p_hcp
        r_shp = np.zeros((n_samples, 2, 4)) + p_shp

        s_all = np.arange(n_samples)

        n_max_cards = np.zeros((n_samples, 2), dtype=int) + n_cards_to_receive

        js = np.zeros(n_samples, dtype=int)
        # Distribute AK
        while True:
            #print("ak_out_i", ak_out_i)
            s_all_r = s_all[js < ak_out_i.shape[1]]
            if len(s_all_r) == 0:
                break

            js_r = js[s_all_r]
            cards = ak_out_i[s_all_r, js_r]
            receivers = distr2_vec(r_shp[s_all_r, :, cards//8], r_hcp[s_all_r], rng)

            can_receive_cards = cards_received[s_all_r, receivers] < n_max_cards[s_all_r, receivers]

            cards_received[s_all_r[can_receive_cards], receivers[can_receive_cards]] += 1
            h1_h2[s_all_r[can_receive_cards], receivers[can_receive_cards], cards[can_receive_cards]] += 1
            # we update stats from bidding_info so lower odds to get next honor card
            # Above we use hcp / 1.2, but here it is fixed to 3
            r_hcp[s_all_r[can_receive_cards], receivers[can_receive_cards]] -= 3 * hcp_reduction_factor
            r_shp[s_all_r[can_receive_cards], receivers[can_receive_cards], cards[can_receive_cards] // 8] -= shp_reduction_factor
            js[s_all_r[can_receive_cards]] += 1

        js = np.zeros(n_samples, dtype=int)
        while True:
            s_all_r = s_all[js < small_out_i.shape[1]]
            if len(s_all_r) == 0:
                break

            js_r = js[s_all_r]
            cards = small_out_i[s_all_r, js_r]
            receivers = distr_vec(r_shp[s_all_r, :, cards//8], rng)

            can_receive_cards = cards_received[s_all_r, receivers] < n_max_cards[s_all_r, receivers]

            cards_received[s_all_r[can_receive_cards], receivers[can_receive_cards]] += 1
            h1_h2[s_all_r[can_receive_cards], receivers[can_receive_cards], cards[can_receive_cards]] += 1
            r_shp[s_all_r[can_receive_cards], receivers[can_receive_cards], cards[can_receive_cards] // 8] -= shp_reduction_factor
            js[s_all_r[can_receive_cards]] += 1

        assert np.sum(h1_h2) == n_samples * np.sum(n_cards_to_receive)

        # Shuffle the samples generated
        # indices = np.arange(n_samples)
        #return h1_h2[indices]
        return h1_h2

    def get_opening_lead_scores(self, auction, vuln, models, hand, opening_lead_card, dealer):
        assert(hand.shape[1] == models.n_cards_play)
        contract = bidding.get_contract(auction)

        level = int(contract[0])
        strain = bidding.get_strain_i(contract)
        doubled = int('X' in contract)
        redbld = int('XX' in contract)

        x = np.zeros((hand.shape[0], 42), dtype=np.float16)
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
        x[:, 10:] = hand

        b = np.zeros((hand.shape[0], 15), dtype=np.float16)

        n_steps = binary.calculate_step_bidding_info(auction)

        binfo_model = models.binfo_model

        handbidding = hand

        # convert the deck if different for play and bidding
        if models.n_cards_play != models.n_cards_bidding:
            handbidding = np.zeros((hand.shape[0], models.n_cards_bidding))
            for i in range(hand.shape[0]):
                handbidding[i] = binary.parse_hand_f(models.n_cards_bidding)(deck52.handxxto52str(hand[i], models.n_cards_play))

        A = binary.get_auction_binary_sampling(n_steps, auction, lead_index, handbidding, vuln, models, models.n_cards_bidding)

        p_hcp, p_shp = binfo_model.model(A)

        b[:, :3] = p_hcp.reshape((-1, n_steps, 3))[:, -1, :].reshape((-1, 3))
        b[:, 3:] = p_shp.reshape((-1, n_steps, 12))[:, -1, :].reshape((-1, 12))

        if (contract[1] == "N"):
            lead_softmax = models.lead_nt_model.model(x, b)
        else:
            lead_softmax = models.lead_suit_model.model(x, b)

        return lead_softmax[:, opening_lead_card]

    def get_bid_scores(self, nesw_i, partner, auction, vuln, sample_hands, models):
        n_steps = binary.calculate_step_bidding(auction)
        if self.verbose:
            print("sample hand", hand_to_str(sample_hands[0]))
            print("n_step", n_steps)
            print("auction", auction)
            print("nesw_i", nesw_i)

        # convert the deck if different for play and bidding
        if models.n_cards_play != models.n_cards_bidding:
            handbidding = np.zeros((sample_hands.shape[0], models.n_cards_bidding))
            for i in range(sample_hands.shape[0]):
                handbidding[i] = binary.parse_hand_f(models.n_cards_bidding)(deck52.handxxto52str(sample_hands[i], models.n_cards_play))
        else:
            handbidding = sample_hands


        A = binary.get_auction_binary_sampling(n_steps, auction, nesw_i, handbidding, vuln, models, models.n_cards_bidding)
        #print("???: ", n_steps, auction, nesw_i, sample_hands, vuln, models)

        X = np.zeros((sample_hands.shape[0], n_steps, A.shape[-1]), dtype=np.float16)
        X[:, :, :] = A

        actual_bids = bidding.get_bid_ids(auction, nesw_i, n_steps)
        if partner:
            sample_bids = models.bidder_model.model_seq(X)[0]
        else:
            sample_bids = models.opponent_model.model_seq(X)[0]
        sample_bids = sample_bids.reshape((sample_hands.shape[0], n_steps, -1))

        min_scores = np.ones(sample_hands.shape[0])

        # We check the bid for each bidding round
        for i in range(n_steps):
            if actual_bids[i] not in (bidding.BID2ID['PAD_START'], bidding.BID2ID['PAD_END']):
                min_scores = np.minimum(min_scores, sample_bids[:, i, actual_bids[i]])
                # for j in range(sample_hands.shape[0]):
                #     if sample_bids[j, i, actual_bids[i]] > 0.1:
                #         print(bidding.ID2BID[actual_bids[i]], j, hand_to_str(sample_hands[j]), sample_bids[j, i, actual_bids[i]])
                #print(bidding.ID2BID[actual_bids[i]], min_scores)
                #min_scores = min_scores + i * sample_bids[:, i, actual_bids[i]]
                #sum += i
        # for j in range(sample_hands.shape[0]):
        #     if min_scores[j] > 0.1:
        #         print(min_scores[j], j, hand_to_str(sample_hands[j]))
        return min_scores

    def init_rollout_states(self, trick_i, player_i, card_players, player_cards_played, shown_out_suits, current_trick, dealer, auction, hand_str, public_hand_str,vuln, models, rng):
        hand_bidding = binary.parse_hand_f(models.n_cards_bidding)(hand_str)
        n_samples = self.sample_hands_play
        contract = bidding.get_contract(auction)
        if self.verbose:
            print(f"Called init_rollout_states {n_samples} - Contract {contract} - Player {player_i}")


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
            if hidden_cards_no < 16:
                sample_boards_for_play = sample_boards_for_play // 4
            if hidden_cards_no < 8:
                sample_boards_for_play = sample_boards_for_play // 10
            # The more cards we know the less samples are needed to 
            h1_h2 = self.shuffle_cards_bidding_info(
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
            print(f"players_states {states[0].shape[0]} trick {trick_i}")

        unique_indices = np.ones(states[0].shape[0]).astype(bool)

        # this is to see how many samples we actually have
        samples = []
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

        # Use the unique_indices to filter player_states
        states = [state[unique_indices] for state in states]
        if self.verbose:
            print(f"Unique states {states[0].shape[0]}")

        accept, c_hcp, c_shp = self.validate_shape_and_hcp_for_sample(auction, known_nesw, hand_bidding, vuln, h_1_nesw, h_2_nesw, hidden_1_i, hidden_2_i, states, models)

        if self.use_bidding_info:
            if np.sum(accept) < n_samples:
                accept = np.ones_like(accept).astype(bool)

            states = [state[accept] for state in states]
        
        if self.verbose:
            print(f"States {states[0].shape[0]} before checking opening lead (after shape and hcp)")

        # reject samples inconsistent with the opening lead
        # We will only check opening lead if we have a lot of samples, as we can't trust other will follow the same lead rules
        if self.lead_accept_threshold > 0:
            states = self.validate_opening_lead_for_sample(trick_i, hidden_1_i, hidden_2_i, current_trick, player_cards_played, models, auction, vuln, states, dealer)
            if self.verbose:
                print(f"States {states[0].shape[0]} after checking lead")
        assert states[0].shape[0] > 0, "No samples after opening lead"

        if self.play_accept_threshold > 0 and trick_i <= 11:
            states = self.validate_play_until_now(trick_i, current_trick, leader_i, player_cards_played, hidden_1_i, hidden_2_i, states, models, contract)
        if self.verbose:
            print(f"States {states[0].shape[0]} after checking the play.")

        assert states[0].shape[0] > 0, "No samples after checking play"
        min_bid_scores = np.ones(states[0].shape[0])

        # Loop the samples for each of the 2 hidden hands to check bidding
        # We should generally trust our partners bidding most
        
        for h_i in [hidden_1_i, hidden_2_i]:
            #if (player_i + 2) % 4 == h_i:
            h_i_nesw = player_to_nesw_i(h_i, contract)
            partner = player_i == (h_i + 2) % 4
            bid_scores = self.get_bid_scores(h_i_nesw, partner, auction, vuln, states[h_i][:, 0, :32], models)
            # print("bid_scores", h_i, bid_scores)
            # for i in range(bid_scores.shape[0]):
            #     if bid_scores[i] > 0.1:
            #         sample = '%s %s %s %s' % (
            #             hand_to_str(states[0][i, 0, :32].astype(int)),
            #             hand_to_str(states[1][i, 0, :32].astype(int)),
            #             hand_to_str(states[2][i, 0, :32].astype(int)),
            #             hand_to_str(states[3][i, 0, :32].astype(int)),
            #         )
            #         print(i, sample, bid_scores[i])
            min_bid_scores = np.minimum(min_bid_scores, bid_scores)
        
    
        # if trick_i == 7:
        # print("min_bid_scores", min_bid_scores)
        # for i in range(min(min_bid_scores.shape[0], 100)):
        #      if min_bid_scores[i] > 0.04:
        #         sample = '%s %s %s %s' % (
        #             hand_to_str(states[0][i, 0, :32].astype(int)),
        #             hand_to_str(states[1][i, 0, :32].astype(int)),
        #             hand_to_str(states[2][i, 0, :32].astype(int)),
        #             hand_to_str(states[3][i, 0, :32].astype(int)),
        #         )
        #         print(sample, min_bid_scores[i])

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
        assert bidding_states[0].shape[0] > 0, "No samples after bidding"

        # Count how many samples we found matching the bidding
        valid_bidding_samples = np.sum(sorted_min_bid_scores > self.bid_accept_play_threshold)
        if self.verbose:
            print("Bidding samples accepted: ",valid_bidding_samples)

        good_quality = True
        # With only few cards left we will not filter the samples according to the bidding.
        if trick_i <= 11:
            # trusting the bidding after sampling cards
            # This could probably be set based on number of deals matching or sorted
            if valid_bidding_samples >= self.sample_hands_play: 
                good_quality = True
                if self.verbose:
                    print("Enough samples above threshold: ",valid_bidding_samples)
                bidding_states = [state[sorted_min_bid_scores > self.bid_accept_play_threshold] for state in bidding_states]
                # Randomize the samples, as we have to many
                # SHould be based on likelyness of how well the bidding match
                random_indices = rng.permutation(bidding_states[0].shape[0])
                bidding_states = [state[random_indices] for state in bidding_states]
                sorted_min_bid_scores = sorted_min_bid_scores[random_indices]
            else:            
                if valid_bidding_samples < self.min_sample_hands_play: 
                    good_quality = False
                    if np.sum(sorted_min_bid_scores > self.bid_extend_play_threshold) == 0:
                        sys.stderr.write(" We have no idea about what the bidding means\n")
                        # We just take top three as we really have no idea about what the bidding means
                        bidding_states = [state[:self.min_sample_hands_play_bad] for state in bidding_states]
                    else:
                        bidding_states = [state[sorted_min_bid_scores > self.bid_extend_play_threshold] for state in bidding_states]
                        good_quality = True
                        # Limit to just the minimum needed
                        bidding_states = [state[:self.min_sample_hands_play] for state in bidding_states]
                else:
                    good_quality = True
                    if self.verbose:
                        print("Enough samples above threshold: ",self.bid_accept_play_threshold)
                    bidding_states = [state[sorted_min_bid_scores > self.bid_accept_play_threshold] for state in bidding_states]
                sorted_min_bid_scores = sorted_min_bid_scores[:bidding_states[0].shape[0]]

        if self.verbose:
            print(f"Returning {min(bidding_states[0].shape[0],n_samples)}")
        assert bidding_states[0].shape[0] > 0, "No samples for DDSolver"
        
        probability_of_occurence = convert_to_probability(sorted_min_bid_scores[:min(bidding_states[0].shape[0],n_samples)])

        return [state[:n_samples] for state in bidding_states], sorted_min_bid_scores[:min(bidding_states[0].shape[0],n_samples)], c_hcp, c_shp, good_quality, probability_of_occurence
    
    def validate_shape_and_hcp_for_sample(self, auction, known_nesw, hand, vuln, h_1_nesw, h_2_nesw, hidden_1_i, hidden_2_i, states, models):
        n_steps = binary.calculate_step_bidding_info(auction)

        A = binary.get_auction_binary_sampling(n_steps, auction, known_nesw, hand, vuln, models, models.n_cards_bidding)

        p_hcp, p_shp = models.binfo_model.model(A)

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
                    accept_shp &= np.sum(states[[hidden_1_i, hidden_2_i][i]][:, 0, (j*8):((j+1)*8)], axis=1) <= np.round(c_shp[i, j]) + 2
                if np.round(c_shp[i, j] >= 5):
                    accept_shp &= np.sum(states[[hidden_1_i, hidden_2_i][i]][:, 0, (j*8):((j+1)*8)], axis=1) >= np.round(c_shp[i, j]) - 1
                if np.round(c_shp[i, j] >= 6):
                    accept_shp &= np.sum(states[[hidden_1_i, hidden_2_i][i]][:, 0, (j*8):((j+1)*8)], axis=1) >= np.round(c_shp[i, j])

        accept = accept_hcp & accept_shp

        return accept, c_hcp, c_shp.flatten()

    def validate_opening_lead_for_sample(self, trick_i, hidden_1_i, hidden_2_i, current_trick, player_cards_played, models, auction, vuln, states, dealer):
        if self.verbose:
            print("validate_opening_lead_for_sample")
            print("lead_accept_threshold", self.lead_accept_threshold)
        # Only make the test if opening leader (0) is hidden
        # The primary idea is to filter away hands, that lead the Q as it denies the K
        if (hidden_1_i == 0 or hidden_2_i == 0) and states[0].shape[0] > self.min_sample_hands_play * 2:
            if (hidden_2_i == 3):
                # We are RHO and trust partners lead
                lead_accept_threshold = self.lead_accept_threshold + self.lead_accept_threshold_partner_trust
                if self.verbose:
                    print(f"RHO and trust partners lead: {lead_accept_threshold:0.3f}")
            else: 
                # How much trust that opponents would have lead the actual card from the hand sampled
                lead_accept_threshold = self.lead_accept_threshold
                if states[0].shape[0] <= self.min_sample_hands_play * 2:
                    return states        
            opening_lead = current_trick[0] if trick_i == 0 else player_cards_played[0][0]
            lead_scores = self.get_opening_lead_scores(auction, vuln, models, states[0][:, 0, :models.n_cards_play], opening_lead, dealer)
            while np.sum(lead_scores >= lead_accept_threshold) < self.min_sample_hands_play and lead_accept_threshold > 0:
                # We are RHO and trust partners lead
                #print("Reducing threshold")
                lead_accept_threshold *= 0.5
                #print(lead_accept_threshold)

            # If we did not find 2 samples we ignore the test for opening lead
            if np.sum(lead_scores >= lead_accept_threshold) > 1:
                states = [state[lead_scores > lead_accept_threshold] for state in states]
                
        return states

    # Check that the play until now is expected with the samples
    # In principle we do this to eliminated hands, where the card played is inconsistent with the sample
    # We should probably only validate partner as he follow our rules (what is in the neural net)
    def validate_play_until_now(self, trick_i, current_trick, leader_i, player_cards_played, hidden_1_i, hidden_2_i, states, models, contract):
        if self.verbose:
            print("Validating play")
            print(trick_i, current_trick, leader_i, player_cards_played, hidden_1_i, hidden_2_i, states[0].shape[0])
        min_scores = np.ones(states[0].shape[0])
        strain_i = bidding.get_strain_i(contract)
        # Select playing models based on suit or NT
        if strain_i != 0:
            playermodelindex = 4
        else:
            playermodelindex = 0

        for p_i in [hidden_1_i, hidden_2_i]:

            if trick_i == 0 and p_i == 0:
                continue
            # We will not test declarer
            if p_i == 3:
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
                return states
            
            if self.verbose:
                print(f"cards_played by {p_i} {cards_played}")
            # We have 11 rounds of play in the neural network, but might have only 10 for Lefty
            if p_i == 0 and not models.opening_lead_included:
                n_tricks_pred = trick_i + len(card_played_current_trick)
            else:
                n_tricks_pred = trick_i + len(card_played_current_trick)
                
            # Depending on suit or NT we must select the right model
            # 0-3 is for NT 4-7 is for suit
            # When the player is instantiated the right model is selected, but here we get it from the configuration
            p_cards = models.player_models[p_i+playermodelindex].model(states[p_i][:, :n_tricks_pred, :])
            card_scores = p_cards[:, np.arange(len(cards_played)), cards_played]
            # The opening lead is validated elsewhere, so we just change the score to 1 for all samples
            if p_i == 0 and models.opening_lead_included:
                card_scores[:, 0] = 1

            min_scores = np.minimum(min_scores, np.min(card_scores, axis=1))
            
        # Trust in the play until now
        play_accept_threshold = self.play_accept_threshold

        if self.verbose:
            print(f"Found deals above play threshold: {np.sum(min_scores > play_accept_threshold)} ")

        while np.sum(min_scores > play_accept_threshold) < self.min_play_accept_threshold_samples and play_accept_threshold > 0:
            play_accept_threshold -= 0.01
            # print(f"play_accept_threshold {play_accept_threshold} reduced")

        s_accepted = min_scores > play_accept_threshold

        states = [state[s_accepted] for state in states]
        return states
    