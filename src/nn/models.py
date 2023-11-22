import os
import os.path

from configparser import ConfigParser

from nn.player import BatchPlayer, BatchPlayerLefty
from nn.bidder import Bidder
from nn.bid_info import BidInfo
from nn.leader import Leader
from nn.lead_singledummy import LeadSingleDummy


class Models:

    def __init__(self, bidder_model, binfo, lead, sd_model, player_models, search_threshold, lead_threshold, no_search_threshold, lead_accept_nn, include_system, ns, ew):
        self.bidder_model = bidder_model
        self.binfo = binfo
        self.lead = lead
        self.sd_model = sd_model
        self.player_models = player_models
        self._lead_threshold = lead_threshold
        self._search_threshold = search_threshold
        self._no_search_threshold = no_search_threshold
        self._lead_accept_nn = lead_accept_nn
        self.include_system = include_system
        self.ns = ns
        self.ew = ew

    @classmethod
    def from_conf(cls, conf: ConfigParser, base_path=None) -> "Models":
        if base_path is None:
            base_path = os.getenv('BEN_HOME') or '..'
        search_threshold = float(conf['bidding']['search_threshold'])
        no_search_threshold = float(conf['bidding']['no_search_threshold'])
        lead_threshold = float(conf['lead']['lead_threshold'])
        lead_accept_nn = float(conf['lead']['lead_accept_nn'])
        include_system = bool(conf['models']['include_system'])
        if include_system == True:
            ns = float(conf['models']['ns'])
            ew = float(conf['models']['ew'])
        return cls(
            bidder_model=Bidder('bidder', os.path.join(base_path, conf['bidding']['bidder'])),
            binfo=BidInfo(os.path.join(base_path, conf['bidding']['info'])),
            lead=Leader(os.path.join(base_path, conf['lead']['lead'])),
            sd_model=LeadSingleDummy(os.path.join(base_path, conf['eval']['lead_single_dummy'])),
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
            include_system=include_system,
            ns=ns,
            ew=ew
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
