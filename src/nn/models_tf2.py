import os
import os.path

from configparser import ConfigParser

from nn.player_tf2 import BatchPlayer, BatchPlayerLefty
from nn.bidder_tf2 import Bidder
from nn.bid_info_tf2 import BidInfo
from nn.leader_tf2 import Leader
from nn.lead_singledummy_tf2 import LeadSingleDummy


class Models:

    def __init__(self, bidder_model, binfo, lead, sd_model, sd_model_no_lead, player_models, search_threshold, lead_threshold, no_search_threshold, lead_accept_nn, ns, ew, model_version):
        self.bidder_model = bidder_model
        self.binfo = binfo
        self.lead = lead
        self.sd_model = sd_model
        self.sd_model_no_lead = sd_model_no_lead
        self.player_models = player_models
        self._lead_threshold = lead_threshold
        self._search_threshold = search_threshold
        self._no_search_threshold = no_search_threshold
        self._lead_accept_nn = lead_accept_nn
        self.ns = ns
        self.ew = ew
        self.model_version = model_version

    @classmethod
    def from_conf(cls, conf: ConfigParser, base_path=None) -> "Models":
        if base_path is None:
            base_path = os.getenv('BEN_HOME') or '..'
        model_version = conf.getint('models','model_version', fallback=1)
        search_threshold = float(conf['bidding']['search_threshold'])
        no_search_threshold = float(conf['bidding']['no_search_threshold'])
        lead_threshold = float(conf['lead']['lead_threshold'])
        lead_accept_nn = float(conf['lead']['lead_accept_nn'])
        use_bba = conf.getboolean('models', 'use_bba', fallback=False)
        if model_version != 0:
            ns = float(conf['models']['ns'])
            ew = float(conf['models']['ew'])
        else:
            ns = -1
            ew = -1
        return cls(
            bidder_model=Bidder('bidder', os.path.join(base_path, conf['bidding']['bidder'])),
            binfo_model=BidInfo(os.path.join(base_path, conf['bidding']['info'])),
            lead_suit_model=Leader(os.path.join(base_path, conf['lead']['lead_suit'])),
            lead_nt_model=Leader(os.path.join(base_path, conf['lead']['lead_nt'])),
            sd_model=LeadSingleDummy(os.path.join(base_path, conf['eval']['lead_single_dummy'])),
            sd_model_no_lead=LeadSingleDummy(os.path.join(base_path, conf['eval']['no_lead_single_dummy'])),
            player_models=[
                BatchPlayerLefty('lefty', os.path.join(base_path, conf['cardplay']['lefty'])),
                BatchPlayer('dummy', os.path.join(base_path, conf['cardplay']['dummy'])),
                BatchPlayer('righty', os.path.join(base_path, conf['cardplay']['righty'])),
                BatchPlayer('decl', os.path.join(base_path, conf['cardplay']['decl']))
            ],
            search_threshold=search_threshold,
            lead_threshold=lead_threshold,
            no_search_threshold=no_search_threshold,
            lead_accept_nn=lead_accept_nn,
            ns=ns,
            ew=ew,
            use_bba=use_bba,
            model_version = model_version
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
    
