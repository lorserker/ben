import os
import os.path

from configparser import ConfigParser
import nn.player as player

from nn.bidder import Bidder
from nn.bid_info import BidInfo
from nn.leader import Leader
from nn.lead_singledummy import LeadSingleDummy


class Models:

    def __init__(self, bidder_model, binfo, lead, sd_model, player_models):
        self.bidder_model = bidder_model
        self.binfo = binfo
        self.lead = lead
        self.sd_model = sd_model
        self.player_models = player_models

    @classmethod
    def from_conf(cls, conf: ConfigParser, base_path) -> "Models":
        return cls(
            bidder_model=Bidder('bidder', os.path.join(
                base_path, conf['bidding']['bidder'])),
            binfo=BidInfo(os.path.join(base_path, conf['bidding']['info'])),
            lead=Leader(os.path.join(base_path, conf['lead']['lead'])),
            sd_model=LeadSingleDummy(os.path.join(
                base_path, conf['eval']['lead_single_dummy'])),
            player_models=[
                player.BatchPlayerLefty('lefty', os.path.join(
                    base_path, conf['cardplay']['lefty'])),
                player.BatchPlayer('dummy', os.path.join(
                    base_path, conf['cardplay']['dummy'])),
                player.BatchPlayer('righty', os.path.join(
                    base_path, conf['cardplay']['righty'])),
                player.BatchPlayer('decl', os.path.join(
                    base_path, conf['cardplay']['decl']))
            ],
        )
