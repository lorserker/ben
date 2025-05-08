import os
import os.path

from configparser import ConfigParser
from nn.bidder_tf2 import Bidder

class Opponents:

    def __init__(self, opponentname, opponent_model, bba_cc, lead_from_pips_nt, lead_from_pips_suit):
        self.name = opponentname
        self.opponent_model = opponent_model
        self.bba_cc = bba_cc
        self.lead_from_pips_nt = lead_from_pips_nt
        self.lead_from_pips_suit = lead_from_pips_suit

    @classmethod
    def from_conf(cls, conf: ConfigParser, base_path=None, verbose=False) -> "Opponents":
        if base_path is None:
            base_path = os.getenv('BEN_HOME') or '..'
        opponentname = conf.get('models', 'opponentname', fallback="BEN")
        lead_from_pips_nt = conf.get('lead', 'opponent_lead_from_pips_nt', fallback="random")
        lead_from_pips_suit = conf.get('lead', 'opponent_lead_from_pips_suit', fallback="random")
        bba_cc =conf.get('models', 'bba_their_cc', fallback=None)
        if bba_cc:
            bba_cc = os.path.join(base_path, bba_cc)
        if verbose:
            print(f"loaded BBA cc {bba_cc}")
        opponent_model = Bidder('opponent', os.path.join(base_path, conf['bidding']['opponent']),alert_supported=False)
        if verbose:
            print(f"loaded opponent model {opponent_model.name}")

        return cls(
            opponentname=opponentname,
            opponent_model=opponent_model,
            bba_cc=bba_cc,
            lead_from_pips_nt=lead_from_pips_nt,
            lead_from_pips_suit=lead_from_pips_suit,
        )

