import os
import os.path

from configparser import ConfigParser
from nn.bidder_tf2 import Bidder

class Opponents:

    def __init__(self, name, model_version, n_cards_bidding, opponent_model, bba_cc, lead_from_pips_nt, lead_from_pips_suit):
        self.name = name
        self.model_version = model_version
        self.n_cards_bidding = n_cards_bidding
        self.opponent_model = opponent_model
        self.bba_cc = bba_cc
        self.lead_from_pips_nt = lead_from_pips_nt
        self.lead_from_pips_suit = lead_from_pips_suit

    @classmethod
    def from_conf(cls, conf: ConfigParser, base_path=None, verbose=False) -> "Opponents":
        if base_path is None:
            base_path = os.getenv('BEN_HOME') or '..'
        name = conf.get('models', 'name', fallback="BEN")
        model_version = conf.getint('models', 'model_version', fallback=2)
        n_cards_bidding = conf.getint('models', 'n_cards_bidding', fallback=32)
        lead_from_pips_nt = conf.get('lead', 'lead_from_pips_nt', fallback="random")
        lead_from_pips_suit = conf.get('lead', 'lead_from_pips_suit', fallback="random")
        bba_cc =conf.get('models', 'bba_our_cc', fallback=None)
        if bba_cc:
            bba_cc = os.path.join(base_path, bba_cc)
        if verbose:
            print(f"loaded BBA cc {bba_cc}")
        opponent_model = Bidder('opponent', os.path.join(base_path, conf['bidding']['bidder']),alert_supported=False)
        if verbose:
            print(f"loaded opponent model {opponent_model.name}")

        return cls(
            name=name,
            model_version=model_version,
            n_cards_bidding=n_cards_bidding,
            opponent_model=opponent_model,
            bba_cc=bba_cc,
            lead_from_pips_nt=lead_from_pips_nt,
            lead_from_pips_suit=lead_from_pips_suit,
        )

