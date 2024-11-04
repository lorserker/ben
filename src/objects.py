import numpy as np

class Card:

    SUITS = 'SHDC'
    RANKS = 'AKQJT98765432'
    X_RANKS = 'AKQJT98x'

    def __init__(self, suit, rank, xcards=False):
        self.suit = suit
        self.rank = rank
        self.xcards = xcards

        self.SUITS = Card.SUITS
        self.RANKS = Card.X_RANKS if xcards else Card.RANKS

    def symbol(self):
        suit_symbol = self.SUITS[self.suit]
        rank_symbol = self.RANKS[self.rank]
        return '{}{}'.format(suit_symbol, rank_symbol)

    def symbol_reversed(self):
        suit_symbol = self.SUITS[self.suit]
        rank_symbol = self.RANKS[self.rank]
        return '{}{}'.format(rank_symbol, suit_symbol)

    def code(self):
        return len(self.RANKS) * self.suit + self.rank

    def __repr__(self):
        return self.symbol()

    def __str__(self):
        return self.symbol()

    @classmethod
    def from_symbol(cls, symbol, xcards=False):
        assert len(symbol) == 2, symbol

        suit_symbol = symbol[0].upper()
        rank_symbol = symbol[1].upper()

        ranks = Card.X_RANKS if xcards else Card.RANKS

        return cls(suit=Card.SUITS.index(suit_symbol), rank=ranks.index(rank_symbol), xcards=xcards)

    @classmethod
    def from_code(cls, code, xcards=False):
        n_ranks = 8 if xcards else 13
        return cls(suit=code // n_ranks, rank=code % n_ranks, xcards=xcards)

class Auction:
    def __init__(self) -> None:
        self.auction = None
        self.contract = None
        self.dealer = None

    def auction_over(self):
        if len(self.auction) < 4:
            return False
        if self.auction[-1] == 'PAD_END':
            return True
        contract = self.last_contract(self.auction)
        if contract is None:
            return all([bid == 'PASS' for bid in self.auction[-4:]]) and all([bid == 'PAD_START' for bid in self.auction[:-4]])
        else:
            return all([bid == 'PASS' for bid in self.auction[-3:]])

    def last_contract(self, auction):
        for bid in reversed(auction):
            if self.is_contract(bid):
                return bid
        return None

    def is_contract(self, bid):
        return bid[0].isdigit()

class CandidateCard:

    def __init__(self, card, insta_score, expected_tricks_sd = None, expected_tricks_dd = None, p_make_contract=None, expected_score_sd=None, expected_score_dd=None, expected_score_mp=None, expected_score_imp=None, msg=None):
        self.card = card
        self.insta_score = None if insta_score is None else float(insta_score)
        self.expected_tricks_sd = None if expected_tricks_sd is None else float(expected_tricks_sd)
        self.expected_tricks_dd = None if expected_tricks_dd is None else float(expected_tricks_dd)
        self.p_make_contract = None if p_make_contract is None else float(p_make_contract)
        self.expected_score_sd = None if expected_score_sd is None else float(expected_score_sd)
        self.expected_score_dd = None if expected_score_dd is None else float(expected_score_dd)
        self.expected_score_mp = None if expected_score_mp is None else float(expected_score_mp)
        self.expected_score_imp = None if expected_score_imp is None else float(expected_score_imp)
        self.msg = msg

    def __str__(self):
        parts = [
            f"card={self.card}",
            f"insta_score={self.insta_score:.4f}" if self.insta_score is not None else None,
            f"exp_tricks_sd={self.expected_tricks_sd:.2f}" if self.expected_tricks_sd is not None else None,
            f"exp_tricks_dd={self.expected_tricks_dd:.4f}" if self.expected_tricks_dd is not None else None,
            f"p_make_contract={self.p_make_contract:.4f}" if self.p_make_contract is not None else None,
            f"exp_score_sd={self.expected_score_sd:.1f}" if self.expected_score_sd is not None else None,
            f"exp_score_dd={self.expected_score_dd:.2f}" if self.expected_score_dd is not None else None,
            f"exp_score_mp={self.expected_score_mp:.0f}" if self.expected_score_mp is not None else None,
            f"exp_score_imp={self.expected_score_imp:.2f}" if self.expected_score_imp is not None else None,
            f"msg={self.msg}" if self.msg is not None else None
        ]
        
        # Filter out None values
        filtered_parts = filter(None, parts)
        
        return f"CandidateCard({', '.join(filtered_parts)})"
    def to_dict(self):
        result = {
            'card': self.card.symbol(),
        }
        if self.insta_score is not None:
            result['insta_score'] = round(self.insta_score,3)
        if self.expected_tricks_sd is not None:
            result['expected_tricks_sd'] = round(self.expected_tricks_sd,2)
        if self.expected_tricks_dd is not None:
            result['expected_tricks_dd'] = round(self.expected_tricks_dd,2)
        if self.p_make_contract is not None:
            result['p_make_contract'] = round(self.p_make_contract,2)
        if self.expected_score_sd is not None:
            result['expected_score_sd'] = round(self.expected_score_sd)
        if self.expected_score_dd is not None:
            result['expected_score_dd'] = round(self.expected_score_dd)
        if self.expected_score_mp is not None:
            result['expected_score_mp'] = self.expected_score_mp
        if self.expected_score_imp is not None:
            result['expected_score_imp'] = self.expected_score_imp
        if self.msg is not None:
            result['msg'] = self.msg

        return result


class CardResp:

    def __init__(self, card, candidates, samples, shape, hcp, quality, who):
        self.card = card
        self.candidates = candidates
        self.samples = samples
        self.shape = shape
        self.hcp = hcp
        self.quality = quality
        self.who = who

    def __str__(self):
        return f"CardResp(card={self.card}, candidates={self.candidates}, samples={self.samples}, shape={self.shape}, hcp={self.hcp}, quality={self.quality}, who={self.who})"
    
    def convert_to_floats(self, array):
        return [round(float(value), 1) if float(value) != int(value) else int(value) for value in array]

    def to_dict(self):
        
        if isinstance(self.hcp, np.ndarray):
            hcp_values = self.convert_to_floats(self.hcp)
        else:
            hcp_values = self.hcp

        if isinstance(self.shape, np.ndarray):
            shape_values = self.convert_to_floats(self.shape)
        elif self.shape is None:
            shape_values = None
        else:
            shape_values = self.shape
          
        result = {
            'card': self.card.symbol(),
            'who' : self.who
        }

        if self.quality is not None:
            result['quality'] = str(round(self.quality,4))
        if hcp_values and hcp_values != -1:
            result['hcp'] = hcp_values
        if shape_values and shape_values != -1:    
            result['shape'] = shape_values
        if len(self.candidates) > 0:
            result['candidates'] = [cand.to_dict() for cand in self.candidates]
        if self.samples != None and len(self.samples) > 0:
            result['samples'] = self.samples

        return result


class CandidateBid:

    def __init__(self, bid, insta_score, expected_score=None, expected_mp=None, expected_imp=None, expected_tricks=None, adjust=None, alert = None):
        self.bid = bid
        self.insta_score = None if insta_score is None else float(insta_score)
        self.expected_score = None if expected_score is None else float(expected_score)
        self.expected_mp = None if expected_mp is None else float(expected_mp)
        self.expected_imp = None if expected_imp is None else float(expected_imp)
        self.expected_tricks = None if expected_tricks is None else float(expected_tricks)
        self.adjust = None if adjust is None else float(adjust)
        self.alert = alert

    def __str__(self):
        bid_str = self.bid.ljust(4) if self.bid is not None else "    "
        insta_score_str = f"{self.insta_score:.4f}" if self.insta_score is not None else "---"
        expected_score_str = f"{self.expected_score:5.0f}" if self.expected_score is not None else "---"
        expected_mp_str = f"{self.expected_mp:5.0f}" if self.expected_mp is not None else "---"
        expected_imp_str = f"{self.expected_imp:5.1f}" if self.expected_imp is not None else "---"
        expected_tricks = f"{self.expected_tricks:5.2f}" if self.expected_tricks is not None else "---"
        adjust_str = f"{self.adjust:4.2f}" if self.adjust is not None else "---"
        alert_str = "alertable" if self.alert else "  "
        return f"CandidateBid(bid={bid_str}, insta_score={insta_score_str}, expected_score={expected_score_str}, expected_mp={expected_mp_str}, expected_imp={expected_imp_str}, expected_tricks={expected_tricks}, adjust={adjust_str}, alert={alert_str})"

    def with_expected_score(self, expected_score, expected_tricks, adjust):
        #print("Updating candidate bid",expected_score, expected_tricks, adjust)
        return CandidateBid(self.bid, self.insta_score, expected_score, self.expected_mp, self.expected_imp, expected_tricks, adjust, self.alert)
    def with_expected_score_mp(self, expected_mp, adjust):
        #print("Updating candidate bid",expected_score, expected_tricks, adjust)
        return CandidateBid(self.bid, self.insta_score, self.expected_score, expected_mp, self.expected_imp, self.expected_tricks, adjust, self.alert)
    def with_expected_score_imp(self, expected_imp, adjust):
        #print("Updating candidate bid",expected_score, expected_tricks, adjust)
        return CandidateBid(self.bid, self.insta_score, self.expected_score, self.expected_mp, expected_imp, self.expected_tricks, adjust, self.alert)

    def to_dict(self):
        result = {
            'call': self.bid,
        }

        if self.insta_score is not None:
            result['insta_score'] = round(self.insta_score,3)

        if self.adjust is not None:
            result['adjustment'] = round(self.adjust)

        if self.expected_score is not None:
            result['expected_score'] = round(self.expected_score)

        if self.expected_mp is not None:
            result['expected_mp'] = self.expected_mp
            result['adjustment'] = round(self.adjust,1)

        if self.expected_imp is not None:
            result['expected_imp'] = self.expected_imp
            result['adjustment'] = round(self.adjust,2)

        if self.expected_tricks is not None:
            result['expected_tricks'] = round(self.expected_tricks,1)        

        if self.alert is not None:
            result['alert'] = str(self.alert)

        return result


class BidResp:

    def __init__(self, bid, candidates, samples, shape, hcp, who, quality, alert, explanation):
        self.bid = bid
        self.candidates = candidates
        self.samples = samples
        self.shape = shape
        self.hcp = hcp
        self.who = who
        self.quality = quality
        self.alert = alert
        self.explanation = explanation
    
    def __str__(self):
        bid_str = self.bid.ljust(4) if self.bid is not None else "    "
        alert_str = ", artificial" if self.alert else "  "
        explain_str = ", " + self.explain if self.explain != None else ""
        return f"BidResp(bid={bid_str}, who={self.who}{alert_str}{explain_str})"

    def convert_to_floats(self, array):
        return [round(float(value), 1) if float(value) != int(value) else int(value) for value in array]

    def to_dict(self):
        
        if isinstance(self.hcp, np.ndarray):
            hcp_values = self.convert_to_floats(self.hcp)
        else:
            hcp_values = self.hcp

        if isinstance(self.shape, np.ndarray):
            shape_values = self.convert_to_floats(self.shape)
        elif self.shape is None:
            shape_values = None
        else:
            shape_values = self.shape

        result = {
            'bid': self.bid,
            'who' : self.who
        }

        if self.quality is not None:
            result['quality'] = str(round(self.quality,2))
        if len(self.candidates) > 0:
            result['candidates'] = [cand.to_dict() for cand in self.candidates]
        if len(self.samples) > 0:
            result['samples'] = self.samples

        if hcp_values and hcp_values != -1:
            result['hcp'] = hcp_values
        if shape_values and shape_values != -1:
            result['shape'] = shape_values

        if self.alert:
            result['alert'] = str(self.alert)
        if self.explanation != None:
            result['explanation'] = str(self.explanation)

        return result

