import os
import os.path

from configparser import ConfigParser
import nn.player as player

from nn.bidder import Bidder
from nn.bid_info import BidInfo
from nn.leader import Leader
from nn.lead_singledummy import LeadSingleDummy


class Models:

    def __init__(self, bidder_model, binfo, lead, sd_model, player_models, search_threshold, lead_threshold):
        self.bidder_model = bidder_model
        self.binfo = binfo
        self.lead = lead
        self.sd_model = sd_model
        self.player_models = player_models
        self._lead_threshold = lead_threshold
        self._search_threshold = search_threshold

    @classmethod
    def from_conf(cls, conf: ConfigParser, base_path=None) -> "Models":
        if base_path is None:
            base_path = os.getenv('BEN_HOME') or '..'
        try:
            search_threshold = float(conf['bidding']['search_threshold'])
        except KeyError:
            # Handle the case where 'search_threshold' key is missing
            search_threshold = 0.10 # default
        try:
            lead_threshold = float(conf['lead']['lead_threshold'])
        except KeyError:
            # Handle the case where 'lead_threshold' key is missing
            lead_threshold = 0.05 # default
        return cls(
            bidder_model=Bidder('bidder', os.path.join(base_path, conf['bidding']['bidder'])),
            binfo=BidInfo(os.path.join(base_path, conf['bidding']['info'])),
            lead=Leader(os.path.join(base_path, conf['lead']['lead'])),
            sd_model=LeadSingleDummy(os.path.join(base_path, conf['eval']['lead_single_dummy'])),
            player_models=[
                player.BatchPlayerLefty('lefty', os.path.join(base_path, conf['cardplay']['lefty'])),
                player.BatchPlayer('dummy', os.path.join(base_path, conf['cardplay']['dummy'])),
                player.BatchPlayer('righty', os.path.join(base_path, conf['cardplay']['righty'])),
                player.BatchPlayer('decl', os.path.join(base_path, conf['cardplay']['decl']))
            ],
            search_threshold=search_threshold,
            lead_threshold=lead_threshold
        )
    
    @property
    def search_threshold(self):
        return self._search_threshold

    @property
    def lead_threshold(self):
        return self._lead_threshold
    
