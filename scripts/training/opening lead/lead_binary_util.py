
import numpy as np
suit_index_lookup = {'S': 0, 'H': 1, 'D': 2, 'C': 3}
seats = ['W', 'N', 'E', 'S']
seat_index = {'W': 0, 'N': 1, 'E': 2, 'S': 3}


def encode_card(card):
    x = np.zeros(32, np.float16)
    if card == '>>':
        return x
    x[get_card_index(card)] = 1
    return x

card_index_lookup = dict(
    zip(
        ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'],
        range(13)
    )
)


card_index_lookup_x = dict(
    zip(
        ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'],
        [0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 7],
    )
)

def get_card_index(card):
    suit, value = card[0], card[1]
    return suit_index_lookup[suit] * 8 + card_index_lookup_x[value]

def convert_auction(auction_str):
    return auction_str.strip().replace('PP', 'PASS').replace('DD', 'X').replace('RR', 'XX').split()


class DealMeta():
    
    def __init__(self, dealer, vuln, level, strain, doubled, redoubled, declarer, tricks_made):
        self.dealer = dealer
        self.vuln = vuln
        self.level = level
        self.strain = strain
        self.doubled = doubled
        self.redoubled = redoubled
        self.declarer = declarer
        self.tricks_made = tricks_made
        
    @classmethod
    def from_str(cls, s):
        #W ALL 3N.-1.N
        parts = s.strip().split()
        outcome = parts[2]
        outcome_parts = outcome.split('.')
        level = int(outcome_parts[0][0])
        doubled = 'X' in outcome_parts[0]
        redoubled = 'XX' in outcome_parts[0]
        strain = outcome_parts[0][1]
        tricks_made = (level + 6) if outcome_parts[1] == '=' else (level + 6) + int(outcome_parts[1])
        declarer = outcome_parts[2]
        
        return cls(
            dealer=parts[0],
            vuln=parts[1],
            level=level,
            strain=strain,
            doubled=doubled,
            redoubled=redoubled,
            declarer=declarer,
            tricks_made=tricks_made
        )
        
    def leader(self):
        return seats[(seat_index[self.declarer] + 1) % 4]
    
    def dealer_relative(self):
        return (seat_index[self.dealer] - seat_index[self.leader()]) % 4
    
    def declarer_vuln(self):
        if self.vuln == 'ALL':
            return True
        if self.vuln == '-':
            return False
        return self.declarer in self.vuln
    
    def leader_vuln(self):
        if self.vuln == 'ALL':
            return True
        if self.vuln == '-':
            return False
        return self.leader() in self.vuln
    
    def get_n_pad_start(self):
        dealer_ix = seat_index[self.dealer]
        declarer_ix = seat_index[self.declarer]
        
        return (dealer_ix - declarer_ix) % 4
    
    def to_dict(self):
        return {
            'dealer': self.dealer,
            'vuln': self.vuln,
            'level': self.level,
            'strain': self.strain,
            'doubled': self.doubled,
            'declarer': self.declarer,
            'tricks_made': self.tricks_made,
            'leader': self.leader(),
            'declarer_vuln': self.declarer_vuln(),
            'leader_vuln': self.leader_vuln()
        }


