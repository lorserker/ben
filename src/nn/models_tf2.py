import os
import os.path

from configparser import ConfigParser

from nn.player_tf2 import BatchPlayer
from nn.bid_info_tf2 import BidInfo
from nn.leader_tf2 import Leader
from nn.lead_singledummy_tf2 import LeadSingleDummy
from nn.contract_tf2 import Contract
from nn.bidder_tf2 import Bidder
from nn.trick_tf2 import Trick

class Models:

    def __init__(self, name, tf_version, model_version, n_cards_bidding, n_cards_play, bidder_model, opponent_model, contract_model, trick_model,binfo_model, lead_suit_model, lead_nt_model, sd_model, sd_model_no_lead, player_models, search_threshold, lead_threshold, 
                 no_search_threshold, eval_after_bid_count, eval_opening_bid,eval_pass_after_bid_count, no_biddingqualitycheck_after_bid_count, min_passout_candidates, min_rescue_reward, min_bidding_trust_for_sample_when_rescue, max_estimated_score,
                 lead_accept_nn, ns, ew, bba_our_cc, bba_their_cc, use_bba, consult_bba, use_bba_rollout, use_bba_to_count_aces, estimator, claim, play_reward_threshold_NN, check_remaining_cards, check_discard, double_dummy, lead_from_pips_nt, lead_from_pips_suit, min_opening_leads, sample_hands_for_review, use_biddingquality, use_biddingquality_in_eval, 
                 double_dummy_calculator, opening_lead_included, use_probability, matchpoint, pimc_verbose, pimc_use_declaring, pimc_use_defending, pimc_use_discarding, pimc_wait, pimc_start_trick_declarer, pimc_start_trick_defender, pimc_stop_trick_declarer, pimc_stop_trick_defender, pimc_constraints, 
                 pimc_constraints_each_trick, pimc_max_playouts, autoplaysingleton, pimc_max_threads, pimc_trust_NN, pimc_ben_dd_declaring, pimc_use_fusion_strategy, pimc_ben_dd_defending, pimc_apriori_probability, 
                 pimc_ben_dd_declaring_weight, pimc_ben_dd_defending_weight, pimc_margin_suit, pimc_margin_hcp, pimc_margin_suit_bad_samples, pimc_margin_hcp_bad_samples, pimc_bidding_quality,
                 alphamju_declaring, alphamju_defending,
                 use_adjustment, adjust_NN, adjust_NN_Few_Samples, adjust_XX, adjust_X, adjust_X_remove, adjust_passout, adjust_passout_negative, adjust_min1, adjust_min2, adjust_min1_by, adjust_min2_by,
                 use_suitc, force_suitc, suitc_sidesuit_check, draw_trump_reward, draw_trump_penalty,       
                 use_real_imp_or_mp, use_real_imp_or_mp_bidding, use_real_imp_or_mp_opening_lead, lead_convention, check_final_contract, max_samples_checked,  
                 alert_supported, alert_threshold,
                 factor_to_translate_to_mp, factor_to_translate_to_imp, use_suit_adjust, suppress_warnings,
                 reward_lead_partner_suit, trump_lead_penalty
                 ):
        self.name = name
        self.tf_version = tf_version
        self.model_version = model_version
        self.n_cards_bidding = n_cards_bidding
        self.n_cards_play = n_cards_play
        self.bidder_model = bidder_model
        self.opponent_model = opponent_model
        self.contract_model = contract_model
        self.trick_model = trick_model
        self.binfo_model = binfo_model
        self.lead_suit_model = lead_suit_model
        self.lead_nt_model = lead_nt_model
        self.sd_model = sd_model
        self.sd_model_no_lead = sd_model_no_lead
        self.player_models = player_models
        self._lead_threshold = lead_threshold
        self._search_threshold = search_threshold
        self._no_search_threshold = no_search_threshold
        self.eval_after_bid_count = eval_after_bid_count
        self.eval_opening_bid = eval_opening_bid
        self.eval_pass_after_bid_count = eval_pass_after_bid_count
        self.no_biddingqualitycheck_after_bid_count = no_biddingqualitycheck_after_bid_count
        self.min_passout_candidates = min_passout_candidates
        self.min_rescue_reward = min_rescue_reward
        self.min_bidding_trust_for_sample_when_rescue = min_bidding_trust_for_sample_when_rescue
        self.max_estimated_score = max_estimated_score
        self._lead_accept_nn = lead_accept_nn
        self.ns = ns
        self.ew = ew
        self.bba_our_cc = bba_our_cc
        self.bba_their_cc = bba_their_cc
        self.use_bba = use_bba
        self.consult_bba = consult_bba
        self.use_bba_rollout = use_bba_rollout
        self.use_bba_to_count_aces = use_bba_to_count_aces
        self.estimator = estimator
        self.claim = claim
        self.play_reward_threshold_NN = play_reward_threshold_NN
        self.check_remaining_cards = check_remaining_cards
        self.check_discard = check_discard
        self.double_dummy = double_dummy
        self.lead_from_pips_nt = lead_from_pips_nt
        self.lead_from_pips_suit = lead_from_pips_suit
        self.min_opening_leads = min_opening_leads
        self.sample_hands_for_review = sample_hands_for_review
        self.use_biddingquality = use_biddingquality
        self.use_biddingquality_in_eval = use_biddingquality_in_eval
        self.double_dummy_calculator = double_dummy_calculator
        self.opening_lead_included = opening_lead_included
        self.pimc_verbose = pimc_verbose
        self.pimc_use_declaring = pimc_use_declaring
        self.pimc_use_defending = pimc_use_defending
        self.pimc_use_discarding = pimc_use_discarding
        self.pimc_wait = pimc_wait
        self.pimc_start_trick_declarer = pimc_start_trick_declarer
        self.pimc_start_trick_defender = pimc_start_trick_defender
        self.pimc_stop_trick_declarer = pimc_stop_trick_declarer
        self.pimc_stop_trick_defender = pimc_stop_trick_defender
        self.pimc_constraints = pimc_constraints
        self.pimc_constraints_each_trick = pimc_constraints_each_trick
        self.pimc_max_playouts = pimc_max_playouts
        self.autoplaysingleton = autoplaysingleton
        self.use_probability = use_probability
        self.matchpoint = matchpoint
        self.pimc_max_threads = pimc_max_threads
        self.pimc_trust_NN = pimc_trust_NN
        self.pimc_ben_dd_declaring = pimc_ben_dd_declaring
        self.pimc_use_fusion_strategy = pimc_use_fusion_strategy
        self.pimc_ben_dd_defending = pimc_ben_dd_defending
        self.pimc_apriori_probability = pimc_apriori_probability
        self.pimc_ben_dd_declaring_weight = pimc_ben_dd_declaring_weight 
        self.pimc_ben_dd_defending_weight = pimc_ben_dd_defending_weight 
        self.pimc_margin_suit = pimc_margin_suit 
        self.pimc_margin_hcp = pimc_margin_hcp 
        self.pimc_margin_suit_bad_samples = pimc_margin_suit_bad_samples 
        self.pimc_margin_hcp_bad_samples = pimc_margin_hcp_bad_samples  
        self.pimc_bidding_quality = pimc_bidding_quality
        self.alphamju_declaring = alphamju_declaring
        self.alphamju_defending = alphamju_defending
        self.use_adjustment = use_adjustment
        self.adjust_NN = adjust_NN
        self.adjust_NN_Few_Samples = adjust_NN_Few_Samples
        self.adjust_XX = adjust_XX
        self.adjust_X = adjust_X
        self.adjust_X_remove = adjust_X_remove
        self.adjust_passout = adjust_passout
        self.adjust_passout_negative = adjust_passout_negative
        self.adjust_min1 = adjust_min1
        self.adjust_min2 = adjust_min2
        self.adjust_min1_by = adjust_min1_by
        self.adjust_min2_by = adjust_min2_by
        self.use_suitc = use_suitc
        self.force_suitc = force_suitc
        self.suitc_sidesuit_check = suitc_sidesuit_check
        self.draw_trump_reward=draw_trump_reward
        self.draw_trump_penalty=draw_trump_penalty
        self.use_real_imp_or_mp = use_real_imp_or_mp
        self.use_real_imp_or_mp_bidding = use_real_imp_or_mp_bidding
        self.use_real_imp_or_mp_opening_lead = use_real_imp_or_mp_opening_lead
        self.lead_convention = lead_convention
        self.check_final_contract = check_final_contract
        self.max_samples_checked = max_samples_checked
        self.alert_supported = alert_supported
        self.alert_threshold = alert_threshold
        self.factor_to_translate_to_mp = factor_to_translate_to_mp
        self.factor_to_translate_to_imp = factor_to_translate_to_imp
        self.use_suit_adjust = use_suit_adjust
        self.suppress_warnings = suppress_warnings
        self.reward_lead_partner_suit = reward_lead_partner_suit
        self.trump_lead_penalty = trump_lead_penalty


    @classmethod
    def from_conf(cls, conf: ConfigParser, base_path=None, verbose=False) -> "Models":
        if base_path is None:
            base_path = os.getenv('BEN_HOME') or '..'
        name = conf.get('models', 'name', fallback="BEN")
        tf_version = conf.getint('models', 'tf_version', fallback=2)
        model_version = conf.getint('models', 'model_version', fallback=2)
        n_cards_bidding = conf.getint('models', 'n_cards_bidding', fallback=32)
        n_cards_play = conf.getint('models', 'n_cards_play', fallback=32)
        alert_supported = conf.getboolean('bidding', 'alert_supported', fallback=False)
        alert_threshold = conf.getfloat('bidding', 'alert_threshold', fallback=0.5)

        search_threshold_str = conf.get('bidding', 'search_threshold', fallback=-1)
        # Check if the value is a list (with brackets), otherwise treat it as a single float
        if search_threshold_str.startswith('[') and search_threshold_str.endswith(']'):
            # Remove brackets and split the string to convert to a list of floats
            search_threshold = [float(x) for x in search_threshold_str.strip('[]').split(',')]
        else:
            # Convert the value to a single float
            search_threshold = float(search_threshold_str)

        no_search_threshold_str = conf.get('bidding', 'no_search_threshold', fallback='1')
        # Check if the value is a list (with brackets), otherwise treat it as a single float
        if no_search_threshold_str.startswith('[') and no_search_threshold_str.endswith(']'):
            # Remove brackets and split the string to convert to a list of floats
            no_search_threshold = [float(x) for x in no_search_threshold_str.strip('[]').split(',')]
        else:
            # Convert the value to a single float
            no_search_threshold = float(no_search_threshold_str)

        eval_after_bid_count = conf.getint('bidding', 'eval_after_bid_count', fallback=-1)
        eval_opening_bid = conf.getboolean('bidding', 'eval_opening_bid', fallback=False)
        eval_pass_after_bid_count = conf.getint('bidding', 'eval_pass_after_bid_count', fallback=-1)
        no_biddingqualitycheck_after_bid_count = conf.getint('bidding', 'no_biddingqualitycheck_after_bid_count', fallback=-1)
        min_passout_candidates = conf.getint('bidding', 'min_passout_candidates', fallback=2)
        min_rescue_reward = conf.getint('bidding', 'min_rescue_reward', fallback=250)
        min_bidding_trust_for_sample_when_rescue = conf.getfloat('bidding','min_bidding_trust_for_sample_when_rescue',fallback=0.5)
        max_estimated_score = conf.getint('bidding', 'max_estimated_score', fallback=300)
        use_biddingquality = conf.getboolean('bidding', 'use_biddingquality', fallback=False)
        check_final_contract = conf.getboolean('bidding', 'check_final_contract', fallback=False)
        max_samples_checked = conf.getint('bidding', 'max_samples_checked', fallback=10)
        use_probability = conf.getboolean('bidding', 'use_probability', fallback=False)
        sample_hands_for_review = conf.getint('sampling', 'sample_hands_for_review', fallback=200)
        lead_threshold = float(conf['lead']['lead_threshold'])
        lead_accept_nn = float(conf['lead']['lead_accept_nn'])
        min_opening_leads = conf.getint('lead', 'min_opening_leads', fallback=1)
        double_dummy = conf.getboolean('lead', 'double_dummy', fallback=False)
        lead_from_pips_nt = conf.get('lead', 'lead_from_pips_nt', fallback="random")
        lead_from_pips_suit = conf.get('lead', 'lead_from_pips_suit', fallback="random")
        matchpoint = conf.getboolean('models', 'matchpoint', fallback=False)
        estimator = conf.get('eval', 'estimator', fallback="sde")
        double_dummy_calculator = conf.getboolean('eval', 'double_dummy_calculator', fallback=False)
        claim = conf.getboolean('cardplay', 'claim', fallback=True)
        play_reward_threshold_NN = conf.getfloat('cardplay', 'play_reward_threshold_NN', fallback=0)
        check_remaining_cards = conf.getint('cardplay', 'check_remaining_cards', fallback=10)
        check_discard = conf.getboolean('cardplay', 'check_discard', fallback=False)
        pimc_verbose = conf.getboolean('pimc', 'pimc_verbose', fallback=True)
        pimc_use_declaring = conf.getboolean('pimc', 'pimc_use_declaring', fallback=False)
        pimc_use_defending = conf.getboolean('pimc', 'pimc_use_defending', fallback=False)
        pimc_use_discarding = conf.getboolean('pimc', 'pimc_use_discarding', fallback=True)
        pimc_wait = conf.getfloat('pimc', 'pimc_wait', fallback=1)
        pimc_start_trick_declarer = conf.getfloat('pimc', 'pimc_start_trick_declarer', fallback=1)
        pimc_start_trick_defender = conf.getfloat('pimc', 'pimc_start_trick_defender', fallback=1)
        pimc_stop_trick_declarer = conf.getfloat('pimc', 'pimc_stop_trick_declarer', fallback=13)
        pimc_stop_trick_defender = conf.getfloat('pimc', 'pimc_stop_trick_defender', fallback=13)
        pimc_max_playouts = conf.getint('pimc', 'pimc_max_playouts', fallback=-1)
        pimc_constraints = conf.getboolean('pimc', 'pimc_constraints', fallback=False)
        pimc_constraints_each_trick = conf.getboolean('pimc', 'pimc_constraints_each_trick', fallback=False)
        autoplaysingleton = conf.getboolean('pimc', 'autoplaysingleton', fallback=False)
        pimc_max_threads = conf.getint('pimc', 'pimc_max_threads', fallback=-1)
        pimc_trust_NN = conf.getfloat('pimc', 'pimc_trust_NN', fallback=0)
        pimc_ben_dd_declaring = conf.getboolean('pimc', 'pimc_ben_dd_declaring', fallback=False)
        pimc_use_fusion_strategy = conf.getboolean('pimc', 'pimc_use_fusion_strategy', fallback=True)
        pimc_ben_dd_defending = conf.getboolean('pimc', 'pimc_ben_dd_defending', fallback=False)
        pimc_apriori_probability = conf.getboolean('pimc', 'pimc_apriori_probability', fallback=False)

        pimc_ben_dd_declaring_weight = conf.getfloat('pimc', 'pimc_ben_dd_declaring_weight', fallback=0.5)
        pimc_ben_dd_defending_weight = conf.getfloat('pimc', 'pimc_ben_dd_defending_weight', fallback=0.5)
        pimc_margin_suit = conf.getint('pimc', 'pimc_margin_suit', fallback=1)
        pimc_margin_hcp = conf.getint('pimc', 'pimc_margin_hcp', fallback=2)
        pimc_margin_suit_bad_samples = conf.getint('pimc', 'pimc_margin_suit_bad_samples', fallback=2)
        pimc_margin_hcp_bad_samples = conf.getint('pimc', 'pimc_margin_hcp_bad_samples', fallback=5)
        pimc_bidding_quality = conf.getfloat('pimc', 'pimc_bidding_quality', fallback=0.4)

        alphamju_declaring = conf.getboolean('alphamju', 'alphamju_declaring', fallback=False)
        alphamju_defending = conf.getboolean('alphamju', 'alphamju_defending', fallback=False)

        use_adjustment = conf.getboolean('adjustments', 'use_adjustment', fallback=True)
        adjust_NN = conf.getint('adjustments', 'adjust_NN', fallback=50)
        adjust_NN_Few_Samples = conf.getint('adjustments', 'adjust_NN_Few_Samples', fallback=500)
        adjust_XX = conf.getint('adjustments', 'adjust_XX', fallback=100)
        adjust_X = conf.getint('adjustments', 'adjust_X', fallback=100)
        adjust_X_remove = conf.getint('adjustments', 'adjust_X_remove', fallback=10)
        adjust_passout = conf.getint('adjustments', 'adjust_passout', fallback=-100)
        adjust_passout_negative = conf.getfloat('adjustments', 'adjust_passout_negative', fallback=1)
        adjust_min1 = conf.getfloat('adjustments', 'adjust_min1', fallback=0.002)
        adjust_min2 = conf.getfloat('adjustments', 'adjust_min2', fallback=0.0002)
        adjust_min1_by = conf.getint('adjustments', 'adjust_min1_by', fallback=200)
        adjust_min2_by = conf.getint('adjustments', 'adjust_min2_by', fallback=200)
        factor_to_translate_to_mp = conf.getint('adjustments', 'factor_to_translate_to_mp', fallback=10)
        factor_to_translate_to_imp = conf.getint('adjustments', 'factor_to_translate_to_imp', fallback=25)
        use_suit_adjust = conf.getboolean('adjustments', 'use_suit_adjust', fallback=False)

        reward_lead_partner_suit = conf.getfloat('adjustments', 'reward_lead_partner_suit', fallback=0)
        trump_lead_penalty_str = conf.get('bidding', 'trump_lead_penalty', fallback=None)
        if trump_lead_penalty_str:
            trump_lead_penalty = [float(x) for x in trump_lead_penalty_str.strip('[]').split(',')]
        else:
            trump_lead_penalty = []
        opening_lead_included = conf.getboolean('cardplay', 'opening_lead_included', fallback=False)
        if not opening_lead_included:
            print("opening lead must be included in TF 2.X models for cardplay")
        use_biddingquality_in_eval = conf.getboolean('cardplay', 'use_biddingquality_in_eval', fallback=False)
        use_suitc = conf.getboolean('cardplay', 'use_suitc', fallback=False)
        force_suitc = conf.getboolean('cardplay', 'force_suitc', fallback=False)
        suitc_sidesuit_check = conf.getboolean('cardplay', 'suitc_sidesuit_check', fallback=False)
        draw_trump_reward = conf.getfloat('cardplay', 'draw_trump_reward', fallback=0.25)
        draw_trump_penalty = conf.getfloat('cardplay', 'draw_trump_penalty', fallback=0.25)
        use_real_imp_or_mp = conf.getboolean('cardplay', 'use_real_imp_or_mp', fallback=False)
        use_real_imp_or_mp_bidding = conf.getboolean('eval', 'use_real_imp_or_mp_bidding', fallback=False)
        use_real_imp_or_mp_opening_lead = conf.getboolean('lead', 'use_real_imp_or_mp_opening_lead', fallback=False)
        lead_convention = conf.getboolean('lead', 'lead_convention', fallback=False)
        suppress_warnings = conf.getboolean('models', 'suppress_warnings', fallback=True)
        use_bba = conf.getboolean('models', 'use_bba', fallback=False)
        consult_bba = conf.getboolean('models', 'consult_bba', fallback=False)
        use_bba_rollout = conf.getboolean('models', 'use_bba_rollout', fallback=False)
        use_bba_to_count_aces = conf.getboolean('models', 'use_bba_to_count_aces', fallback=False)
        bba_our_cc =conf.get('models', 'bba_our_cc', fallback=None)
        if bba_our_cc:
            bba_our_cc = os.path.join(base_path, bba_our_cc)
        bba_their_cc =conf.get('models', 'bba_their_cc', fallback=None)
        if bba_their_cc:
            bba_their_cc = os.path.join(base_path, bba_their_cc)
        if verbose:
            print(f"loaded bba_our_cc and bba_their_cc as {bba_our_cc} and {bba_their_cc}")
        player_names = ['lefty_nt', 'dummy_nt', 'righty_nt', 'decl_nt', 'lefty_suit', 'dummy_suit', 'righty_suit', 'decl_suit']
        ns = int(conf['models']['ns'])
        ew = int(conf['models']['ew'])
        bidder_model = Bidder('bidder', os.path.join(base_path, conf['bidding']['bidder']),alert_supported=alert_supported)
        if conf.has_section('bidding') and conf.get('bidding', 'opponent', fallback=None) not in ('none', None):
            opponent_model = Bidder('opponent', os.path.join(base_path, conf['bidding']['opponent']),alert_supported=alert_supported)
        else:
            opponent_model = Bidder('opponent', os.path.join(base_path, conf['bidding']['bidder']),alert_supported=alert_supported)
        if verbose:
            print(f"Loaded bidding models")
        contract_model=Contract(os.path.join(base_path, conf['contract']['contract']))
        trick_model=Trick(os.path.join(base_path, conf['contract']['trick']))
        binfo_model=BidInfo(os.path.join(base_path, conf['bidding']['info']))
        if verbose:
            print(f"Loaded contract and bidding info models")

        lead_suit_model=Leader(os.path.join(base_path, conf['lead']['lead_suit']))
        lead_nt_model=Leader(os.path.join(base_path, conf['lead']['lead_nt']))
        if verbose:
            print(f"Loaded lead models")
        sd_model=LeadSingleDummy(os.path.join(base_path, conf['eval']['single_dummy_estimator']))
        sd_model_no_lead=LeadSingleDummy(os.path.join(base_path, conf['eval']['double_dummy_estimator']))
        if verbose:
            print(f"Loaded single dummy models")

        player_models=[
            BatchPlayer(name, os.path.join(base_path, conf['cardplay'][name])) for name in player_names
        ]

        if verbose:
            print(f"loaded {len(player_models)} player models")

        return cls(
            name=name,
            tf_version=tf_version,
            model_version=model_version,
            n_cards_bidding=n_cards_bidding,
            n_cards_play=n_cards_play,
            bidder_model=bidder_model,
            opponent_model=opponent_model,
            contract_model=contract_model,
            trick_model=trick_model,
            binfo_model=binfo_model,
            lead_suit_model=lead_suit_model,
            lead_nt_model=lead_nt_model,
            sd_model=sd_model,
            sd_model_no_lead=sd_model_no_lead,
            player_models=player_models,
            search_threshold=search_threshold,
            lead_threshold=lead_threshold,
            no_search_threshold=no_search_threshold,
            eval_after_bid_count=eval_after_bid_count,
            eval_opening_bid=eval_opening_bid,
            eval_pass_after_bid_count=eval_pass_after_bid_count,
            no_biddingqualitycheck_after_bid_count=no_biddingqualitycheck_after_bid_count,
            min_passout_candidates = min_passout_candidates,
            min_rescue_reward = min_rescue_reward,
            min_bidding_trust_for_sample_when_rescue = min_bidding_trust_for_sample_when_rescue,
            max_estimated_score = max_estimated_score,
            lead_accept_nn=lead_accept_nn,
            ns=ns,
            ew=ew,
            bba_our_cc=bba_our_cc,
            bba_their_cc=bba_their_cc,
            use_bba=use_bba,
            consult_bba=consult_bba,
            use_bba_rollout=use_bba_rollout,
            use_bba_to_count_aces=use_bba_to_count_aces,
            estimator=estimator,
            claim=claim,
            play_reward_threshold_NN=play_reward_threshold_NN,
            check_remaining_cards=check_remaining_cards,
            check_discard=check_discard,
            double_dummy=double_dummy,
            lead_from_pips_nt=lead_from_pips_nt,
            lead_from_pips_suit=lead_from_pips_suit,
            min_opening_leads=min_opening_leads,
            sample_hands_for_review=sample_hands_for_review,
            use_biddingquality=use_biddingquality,
            use_biddingquality_in_eval=use_biddingquality_in_eval,
            double_dummy_calculator=double_dummy_calculator,
            opening_lead_included=opening_lead_included,
            use_probability=use_probability,
            alert_supported=alert_supported,
            alert_threshold=alert_threshold,
            matchpoint=matchpoint,
            pimc_verbose=pimc_verbose,
            pimc_use_declaring=pimc_use_declaring,
            pimc_use_defending=pimc_use_defending,
            pimc_use_discarding=pimc_use_discarding,
            pimc_wait=pimc_wait,
            pimc_start_trick_declarer=pimc_start_trick_declarer,
            pimc_start_trick_defender=pimc_start_trick_defender,
            pimc_stop_trick_declarer=pimc_stop_trick_declarer,
            pimc_stop_trick_defender=pimc_stop_trick_defender,
            pimc_constraints=pimc_constraints,
            pimc_constraints_each_trick=pimc_constraints_each_trick,
            pimc_max_playouts=pimc_max_playouts,
            autoplaysingleton=autoplaysingleton,
            pimc_max_threads=pimc_max_threads,
            pimc_trust_NN=pimc_trust_NN,
            pimc_ben_dd_declaring=pimc_ben_dd_declaring,
            pimc_use_fusion_strategy=pimc_use_fusion_strategy,
            pimc_ben_dd_defending=pimc_ben_dd_defending,
            pimc_apriori_probability=pimc_apriori_probability,
            pimc_ben_dd_declaring_weight = pimc_ben_dd_declaring_weight,
            pimc_ben_dd_defending_weight = pimc_ben_dd_defending_weight,
            pimc_margin_suit = pimc_margin_suit,
            pimc_margin_hcp = pimc_margin_hcp,
            pimc_margin_suit_bad_samples = pimc_margin_suit_bad_samples,
            pimc_margin_hcp_bad_samples = pimc_margin_hcp_bad_samples, 
            pimc_bidding_quality = pimc_bidding_quality,
            alphamju_declaring=alphamju_declaring,
            alphamju_defending=alphamju_defending,
            use_adjustment=use_adjustment,
            adjust_NN=adjust_NN,
            adjust_NN_Few_Samples=adjust_NN_Few_Samples,
            adjust_XX=adjust_XX,
            adjust_X=adjust_X,
            adjust_X_remove=adjust_X_remove,
            adjust_passout=adjust_passout,
            adjust_passout_negative=adjust_passout_negative,
            adjust_min1=adjust_min1,
            adjust_min2=adjust_min2,
            adjust_min1_by=adjust_min1_by,
            adjust_min2_by=adjust_min2_by,
            factor_to_translate_to_mp=factor_to_translate_to_mp,
            factor_to_translate_to_imp=factor_to_translate_to_imp,
            use_suit_adjust=use_suit_adjust,
            use_suitc=use_suitc,
            force_suitc=force_suitc,
            suitc_sidesuit_check=suitc_sidesuit_check,
            draw_trump_reward=draw_trump_reward,
            draw_trump_penalty=draw_trump_penalty,
            use_real_imp_or_mp=use_real_imp_or_mp,
            use_real_imp_or_mp_bidding=use_real_imp_or_mp_bidding,
            use_real_imp_or_mp_opening_lead=use_real_imp_or_mp_opening_lead,
            lead_convention = lead_convention,
            check_final_contract=check_final_contract,
            max_samples_checked=max_samples_checked,
            suppress_warnings=suppress_warnings,
            reward_lead_partner_suit=reward_lead_partner_suit,
            trump_lead_penalty=trump_lead_penalty
        )

    @property
    def search_threshold(self):
        return self._search_threshold

    @search_threshold.setter
    def search_threshold(self, value):
        self._search_threshold = value

    @property
    def no_search_threshold(self):
        return self._no_search_threshold

    @no_search_threshold.setter
    def no_search_threshold(self, value):
        self._no_search_threshold = value

    @property
    def lead_threshold(self):
        return self._lead_threshold

    @property
    def lead_accept_nn(self):
        return self._lead_accept_nn
