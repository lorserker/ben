"""
Per-request bidding systems and carding for the API (the Seat Robot API's
ourSystem / theirSystem / ourCarding / theirCarding parameters).

A server plays what it was started with. These parameters override that for one
request only, so one gameapi process can sit at tables with different opponents.
The overrides are applied to a shallow copy of the shared Models, never to the
shared object, because the API serves several requests at once.

What can change per request:
- ourSystem: the bidding and bidding-info networks used for us and partner, and
  our BBA convention card.
- theirSystem: the network used to model the opponents' bidding, and their BBA card.
- ourCarding: the rule for which pip BEN leads (see carding.select_right_card).
- theirCarding: accepted, but BEN reads every opening lead with one fixed table
  (openinglead.validate_lead) and has no signalling logic, so it changes nothing yet.
"""
import copy
import os
from configparser import ConfigParser

from nn.models_tf2 import Bidder, BidInfo
from nn.model_path_provider import IsolatedModelPathProvider, find_best_temp_drive

# Opening-lead pip rules (lead_from_pips_nt, lead_from_pips_suit) for each carding name
CARDINGS = {
    "STANDARD": ("24", "24"),       # 4th best from an honour, 2nd highest from a weak suit
    "STANDARD-35": ("24", "135"),   # as STANDARD, but 3rd/5th best against suit contracts
}

PARAMETERS = ("ourSystem", "theirSystem", "ourCarding", "theirCarding")


class UnknownName(ValueError):
    """A request named a system or carding this server does not know (answered with 400)."""


class System:
    def __init__(self, name, cc, plays, bidder_model, info_model):
        self.name = name
        self.cc = cc
        self.plays = plays
        self.bidder_model = bidder_model
        self.info_model = info_model


class Systems:
    def __init__(self, systems, defaults):
        self.systems = systems
        self.defaults = defaults

    @classmethod
    def load(cls, filename, configuration, models, base_path, verbose=False):
        """Load every system in filename. Networks the startup models already use are
        shared rather than loaded twice."""
        def key(path):
            return os.path.normcase(os.path.normpath(os.path.join(base_path, path)))

        bidder_path = configuration.get('bidding', 'bidder')
        opponent_path = configuration.get('bidding', 'opponent', fallback='none')
        if opponent_path == 'none':
            opponent_path = bidder_path
        loaded = {
            key(bidder_path): models.bidder_model,
            key(opponent_path): models.opponent_model,
            key(configuration.get('bidding', 'info')): models.binfo_model,
        }
        alert_supported = configuration.getboolean('bidding', 'alert_supported', fallback=False)
        path_provider = IsolatedModelPathProvider(
            base_temp_storage_path=find_best_temp_drive(temp_subdir_name="BEN_model_temp_storage"))

        def load_model(path, make):
            if key(path) not in loaded:
                loaded[key(path)] = make(path_provider.get_path(os.path.join(base_path, path)))
            return loaded[key(path)]

        conf = ConfigParser()
        if not conf.read(filename):
            raise FileNotFoundError(f"System list not found: {filename}")
        systems = {}
        for name in conf.sections():
            section = conf[name]
            cc = os.path.join(base_path, section['cc'])
            if not os.path.isfile(cc):
                raise FileNotFoundError(f"System {name}: convention card not found: {cc}")
            plays = section.getboolean('plays', fallback=False)
            bidder_model = load_model(section['bidder'], lambda p: Bidder('bidder', p, alert_supported=alert_supported))
            info_model = load_model(section['info'], BidInfo) if plays else None
            systems[name.upper()] = System(name.upper(), cc, plays, bidder_model, info_model)

        # The names of what the server plays when a request does not say
        def find(path, cc):
            for system in systems.values():
                if system.bidder_model is loaded.get(key(path)) and key(system.cc) == key(cc or ''):
                    return system.name
            return None
        defaults = {
            "ourSystem": find(bidder_path, configuration.get('models', 'bba_our_cc', fallback=None)),
            "theirSystem": find(opponent_path, configuration.get('models', 'bba_their_cc', fallback=None)),
        }
        pips = (models.lead_from_pips_nt, models.lead_from_pips_suit)
        defaults["ourCarding"] = next((name for name, rule in CARDINGS.items() if rule == pips), defaults["ourSystem"])
        defaults["theirCarding"] = defaults["theirSystem"]
        defaults = {k: v for k, v in defaults.items() if v is not None}
        if verbose:
            print(f"Loaded {len(systems)} systems, defaults {defaults}")
        return cls(systems, defaults)

    def capabilities(self):
        """The GET / body: the names a caller may send, and what is played without them."""
        return {
            "playsSystems": [s.name for s in self.systems.values() if s.plays],
            "readsSystems": list(self.systems),
            "cardings": list(self.systems) + list(CARDINGS),
            "defaults": self.defaults,
        }

    def for_request(self, models, args):
        """A copy of models with this request's systems and carding. Raises UnknownName,
        naming the parameter, for a name this server does not know."""
        requested = {p: args.get(p, "").strip().upper() for p in PARAMETERS}
        if not any(requested.values()):
            return copy.copy(models)
        request_models = copy.copy(models)

        if requested["ourSystem"]:
            system = self._system("ourSystem", requested["ourSystem"], must_play=True)
            request_models.bidder_model = system.bidder_model
            request_models.binfo_model = system.info_model
            request_models.bba_our_cc = system.cc
        if requested["theirSystem"]:
            system = self._system("theirSystem", requested["theirSystem"], must_play=False)
            request_models.opponent_model = system.bidder_model
            request_models.bba_their_cc = system.cc
        for parameter in ("ourCarding", "theirCarding"):
            name = requested[parameter]
            if name and name not in CARDINGS and name not in self.systems:
                raise UnknownName(f"Unknown {parameter} '{args.get(parameter)}'. This server knows: {', '.join(self.capabilities()['cardings'])}")
        if requested["ourCarding"] in CARDINGS:
            request_models.lead_from_pips_nt, request_models.lead_from_pips_suit = CARDINGS[requested["ourCarding"]]
        return request_models

    def _system(self, parameter, name, must_play):
        system = self.systems.get(name)
        if system is None or (must_play and not system.plays):
            names = [s.name for s in self.systems.values() if s.plays or not must_play]
            raise UnknownName(f"Unknown {parameter} '{name}'. This server {'plays' if must_play else 'reads'}: {', '.join(names)}")
        return system
