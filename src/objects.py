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

    def code(self):
        return len(self.RANKS) * self.suit + self.rank

    def __repr__(self):
        return self.symbol()

    def __str__(self):
        return self.symbol()

    @classmethod
    def from_symbol(cls, symbol, xcards=False):
        assert len(symbol) == 2

        suit_symbol = symbol[0].upper()
        rank_symbol = symbol[1].upper()

        ranks = Card.X_RANKS if xcards else Card.RANKS

        return cls(suit=Card.SUITS.index(suit_symbol), rank=ranks.index(rank_symbol), xcards=xcards)

    @classmethod
    def from_code(cls, code, xcards=False):
        n_ranks = 8 if xcards else 13
        return cls(suit=code // n_ranks, rank=code % n_ranks, xcards=xcards)



class CandidateCard:

    def __init__(self, card, insta_score, expected_tricks, p_make_contract=None, expected_score=None):
        self.card = card
        self.insta_score = None if insta_score is None else float(insta_score)
        self.expected_tricks = None if expected_tricks is None else float(expected_tricks)
        self.p_make_contract = None if p_make_contract is None else float(p_make_contract)
        self.expected_score = None if expected_score is None else float(expected_score)

    def to_dict(self):
        result = {
            'card': self.card.symbol(),
        }
        if self.insta_score is not None:
            result['insta_score'] = round(self.insta_score,4)
        if self.expected_tricks is not None:
            result['expected_tricks'] = round(self.expected_tricks,2)
        if self.p_make_contract is not None:
            result['p_make_contract'] = round(self.p_make_contract,2)
        if self.expected_score is not None:
            result['expected_score'] = round(self.expected_score)

        return result


class CardResp:

    def __init__(self, card, candidates, samples):
        self.card = card
        self.candidates = candidates
        self.samples = samples

    def to_dict(self):
        return {
            'card': self.card.symbol(),
            'candidates': [cand.to_dict() for cand in self.candidates],
            'samples': self.samples
        }


class CandidateBid:

    def __init__(self, bid, insta_score, expected_score=None, adjust=None):
        self.bid = bid
        self.insta_score = None if insta_score is None else float(insta_score)
        self.expected_score = None if expected_score is None else float(expected_score)
        self.adjust = None if adjust is None else float(adjust)

    def __str__(self):
        return f"CandidateBid(bid={self.bid}, insta_score={self.insta_score}, expected_score={self.expected_score}, adjust={self.adjust})"        

    def with_expected_score(self, expected_score, adjust):
        return CandidateBid(self.bid, self.insta_score, expected_score, adjust)

    def to_dict(self):
        result = {
            'call': self.bid,
        }

        if self.insta_score is not None:
            result['insta_score'] = round(self.insta_score,4)

        if self.expected_score is not None:
            result['expected_score'] = round(self.expected_score)
        
        if self.adjust is not None:
            result['adjustment'] = round(self.adjust)
        
        return result


class BidResp:

    def __init__(self, bid, candidates, samples):
        self.bid = bid
        self.candidates = candidates
        self.samples = samples

    def to_dict(self):
        return {
            'bid': self.bid,
            'candidates': [candidate.to_dict() for candidate in self.candidates],
            'samples': self.samples
        }

