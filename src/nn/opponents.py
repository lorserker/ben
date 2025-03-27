import os
import os.path

from configparser import ConfigParser
from nn.bidder_tf2 import Bidder

class Opponents:

    def __init__(self, name, model_version, n_cards_bidding, opponent_model, bba_their_cc, lead_from_pips_nt, lead_from_pips_suit):
        self.name = name
        self.model_version = model_version
        self.n_cards_bidding = n_cards_bidding
        self.opponent_model = opponent_model
        self.bba_their_cc = bba_their_cc
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
        bba_their_cc =conf.get('models', 'bba_their_cc', fallback=None)
        if bba_their_cc:
            bba_their_cc = os.path.join(base_path, bba_their_cc)
        if verbose:
            print(f"loaded bba_our_cc and bba_their_cc as {bba_our_cc} and {bba_their_cc}")
        opponent_model = Bidder('opponent', os.path.join(base_path, conf['bidding']['opponent']),alert_supported=False)

        return cls(
            name=name,
            model_version=model_version,
            n_cards_bidding=n_cards_bidding,
            opponent_model=opponent_model,
            bba_their_cc=bba_their_cc,
            lead_from_pips_nt=lead_from_pips_nt,
            lead_from_pips_suit=lead_from_pips_suit,
        )

