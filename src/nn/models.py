import os
import os.path

from configparser import ConfigParser

from nn.player import BatchPlayer, BatchPlayerLefty
from nn.bidder import Bidder
from nn.bid_info import BidInfo
from nn.leader import Leader
from nn.lead_singledummy import LeadSingleDummy


class Models:

    def __init__(self, bidder_model, binfo_model, lead_suit_model, lead_nt_model, sd_model, sd_model_no_lead, player_models, search_threshold, lead_threshold, no_search_threshold, eval_after_bid_count, lead_accept_nn, include_system, ns, ew, bba_ns, bba_ew, sameforboth, use_bba, lead_included, claim, double_dummy, min_opening_leads, sample_hands_for_review,use_biddingquality,use_biddingquality_in_eval, double_dummy_eval, include_opening_lead, use_probability, matchpoint, pimc_use, pimc_wait,pimc_start_trick, pimc_hcp_constraints):
        self.bidder_model = bidder_model
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
        self._lead_accept_nn = lead_accept_nn
        self.include_system = include_system
        self.ns = ns
        self.ew = ew
        self.bba_ns = bba_ns
        self.bba_ew = bba_ew
        self.sameforboth = sameforboth
        self.use_bba = use_bba
        self.lead_included = lead_included
        self.claim = claim
        self.double_dummy = double_dummy
        self.min_opening_leads = min_opening_leads
        self.sample_hands_for_review = sample_hands_for_review
        self.use_biddingquality = use_biddingquality
        self.use_biddingquality_in_eval = use_biddingquality_in_eval
        self.double_dummy_eval = double_dummy_eval
        self.include_opening_lead = include_opening_lead
        self.pimc_use = pimc_use
        self.pimc_wait = pimc_wait
        self.pimc_start_trick = pimc_start_trick
        self.pimc_hcp_constraints = pimc_hcp_constraints
        self.use_probability = use_probability
        self.matchpoint = matchpoint

    @classmethod
    def from_conf(cls, conf: ConfigParser, base_path=None) -> "Models":
        if base_path is None:
            base_path = os.getenv('BEN_HOME') or '..'
        search_threshold = float(conf['bidding']['search_threshold'])
        no_search_threshold = conf.getfloat('bidding','no_search_threshold', fallback=1)
        eval_after_bid_count = conf.getint('bidding', 'eval_after_bid_count', fallback=12)
        use_biddingquality = conf.getboolean('bidding', 'use_biddingquality', fallback=False)
        use_probability = conf.getboolean('bidding', 'use_probability', fallback=False)
        sample_hands_for_review = conf.getint('sampling', 'sample_hands_for_review', fallback=200)
        lead_threshold = float(conf['lead']['lead_threshold'])
        lead_accept_nn = float(conf['lead']['lead_accept_nn'])
        min_opening_leads = conf.getint('lead','min_opening_leads', fallback=1)
        double_dummy = conf.getboolean('lead', 'double_dummy', fallback=False)
        include_system = conf.getboolean('models', 'include_system', fallback=False)
        sameforboth = conf.getboolean('models', 'sameforboth', fallback=False)
        use_bba = conf.getboolean('models', 'use_bba', fallback=False)
        matchpoint = conf.getboolean('models', 'matchpoint', fallback=False)
        lead_included = conf.getboolean('eval', 'lead_included', fallback=True)
        double_dummy_eval = conf.getboolean('eval', 'double_dummy_eval', fallback=False)
        claim = conf.getboolean('cardplay', 'claim', fallback=True)
        pimc_use = conf.getboolean('cardplay', 'pimc_use', fallback=False)
        pimc_wait = conf.getfloat('cardplay','pimc_wait', fallback=1)
        pimc_start_trick = conf.getfloat('cardplay','pimc_start_trick', fallback=1)
        pimc_hcp_constraints = conf.getboolean('cardplay', 'pimc_hcp_constraints', fallback=False)
        include_opening_lead = conf.getboolean('cardplay', 'include_opening_lead', fallback=False)
        use_biddingquality_in_eval = conf.getboolean('cardplay', 'claim', fallback=False)
        if include_system == True:
            ns = float(conf['models']['ns'])
            ew = float(conf['models']['ew'])
        else:
            ns = -1
            ew = -1
        if use_bba == True:
            bba_ns = float(conf['models']['bba_ns'])
            bba_ew = float(conf['models']['bba_ew'])
        else:
            bba_ns = -1
            bba_ew = -1
        player_names = ['lefty_nt', 'dummy_nt', 'righty_nt', 'decl_nt', 'lefty_suit', 'dummy_suit', 'righty_suit', 'decl_suit']
        return cls(
            bidder_model=Bidder('bidder', os.path.join(base_path, conf['bidding']['bidder'])),
            binfo_model=BidInfo(os.path.join(base_path, conf['bidding']['info'])),
            lead_suit_model=Leader(os.path.join(base_path, conf['lead']['lead_suit'])),
            lead_nt_model=Leader(os.path.join(base_path, conf['lead']['lead_nt'])),
            sd_model=LeadSingleDummy(os.path.join(base_path, conf['eval']['lead_single_dummy'])),
            sd_model_no_lead=LeadSingleDummy(os.path.join(base_path, conf['eval']['no_lead_single_dummy'])),

            player_models = [
                BatchPlayerLefty(name, os.path.join(base_path, conf['cardplay'][name])) if 'lefty' in name and include_opening_lead == False else
                BatchPlayer(name, os.path.join(base_path, conf['cardplay'][name]))
                for name in player_names
            ],

            search_threshold=search_threshold,
            lead_threshold=lead_threshold,
            no_search_threshold=no_search_threshold,
            eval_after_bid_count=eval_after_bid_count,
            lead_accept_nn=lead_accept_nn,
            include_system=include_system,
            ns=ns,
            ew=ew,
            bba_ns=bba_ns,
            bba_ew=bba_ew,
            sameforboth=sameforboth,
            use_bba=use_bba,
            lead_included=lead_included,
            claim=claim,
            double_dummy=double_dummy,
            min_opening_leads=min_opening_leads,
            sample_hands_for_review=sample_hands_for_review,
            use_biddingquality=use_biddingquality,
            use_biddingquality_in_eval=use_biddingquality_in_eval,
            double_dummy_eval=double_dummy_eval,
            include_opening_lead = include_opening_lead,
            use_probability = use_probability,
            matchpoint = matchpoint,
            pimc_use = pimc_use,
            pimc_wait = pimc_wait,
            pimc_start_trick= pimc_start_trick,
            pimc_hcp_constraints = pimc_hcp_constraints
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
