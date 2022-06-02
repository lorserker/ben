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
    def load(cls, models_dir):
        return cls(
            bidder_model=Bidder('gib21', f'{models_dir}/gib21_model/gib21-1000000'),
            binfo=BidInfo(f'{models_dir}/gib21_info_model/gib21_info-500000'),
            lead=Leader(f'{models_dir}/lead_model_b/lead-1000000'),
            sd_model=LeadSingleDummy(f'{models_dir}/lr3_model/lr3-1000000'),
            player_models=[
                player.BatchPlayerLefty('lefty', f'{models_dir}/lefty_model/lefty-1000000'),
                player.BatchPlayer('dummy', f'{models_dir}/dummy_model/dummy-920000'),
                player.BatchPlayer('righty', f'{models_dir}/righty_model/righty-1000000'),
                player.BatchPlayer('decl', f'{models_dir}/decl_model/decl-1000000')
            ],
        )

