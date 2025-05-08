import numpy as np

card_index_lookup = dict(
    zip(
        ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'],
        range(13)
    )
)

def play_data_iterator(fin):
    lines = []
    for i, line in enumerate(fin):
        line = line.strip()
        if i % 4 == 0 and i > 0:
            yield (lines[0], lines[1], lines[3])
            lines = []

        lines.append(line)

    yield (lines[0], lines[1], lines[3])

def get_cards(play_str):
    cards = []
    i = 0
    while i < len(play_str):
        cards.append(play_str[i:i+2])
        i += 2
    return cards

def get_tricks(cards):
    return list(map(list, np.array(cards).reshape((13, 4))))


def wins_trick_index(trick, trump, lead_index):
    led_suit = trick[0][0]
    card_values = []
    for card in trick:
        suit, value = card[0], 14 - card_index_lookup[card[1]]
        if suit == trump:
            card_values.append(value + 13)
        elif suit == led_suit:
            card_values.append(value)
        else:
            card_values.append(0)
    return (np.argmax(card_values) + lead_index) % 4


def get_play_labels(play_str, trump, player_turn_i):
    tricks = get_tricks(get_cards(play_str))
    
    trick_ix, leads, last_tricks, cards_in, labels = [], [], [], [], []
    
    lead_index = 0
    prev_lead_index = 0
    last_trick = ['>>', '>>', '>>', '>>']
    for trick_i, trick in enumerate(tricks):
        last_tricks.append(last_trick)
        leads.append(prev_lead_index)
        
        current_trick = ['>>', '>>', '>>']
        
        for i, card in enumerate(trick):
            player_i = (lead_index + i) % 4
            
            if player_i == player_turn_i: # the player for whom we generate data is on play
                labels.append(card)
                trick_ix.append(trick_i)
                cards_in.append(current_trick)
                break
            else:
                current_trick.append(card)
                del current_trick[0]

        if lead_index == 0:
            last_trick = trick
        elif lead_index == 1:
            last_trick = trick[3:] + trick[:3]
        elif lead_index == 2:
            last_trick = trick[2:] + trick[:2]
        else:
            last_trick = trick[1:] + trick[:1]
        prev_lead_index, lead_index = lead_index, wins_trick_index(trick, trump, lead_index)
        
    return trick_ix, leads, last_tricks, cards_in, labels
    
suit_index_lookup = {'S': 0, 'H': 1, 'D': 2, 'C': 3}
seats = ['W', 'N', 'E', 'S']
seat_index = {'W': 0, 'N': 1, 'E': 2, 'S': 3}


def hot_encode_card(card):
    x = np.zeros(32, np.float16)
    if card == '>>':
        return x
    x[get_card_index(card)] = 1
    return x


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

def binary_hand(suits):
    x = np.zeros(32, np.float16)
    assert(len(suits) == 4)
    for suit_index in [0, 1, 2, 3]:
        for card in suits[suit_index]:
            card_index = card_index_lookup_x[card]
            x[suit_index * 8 + card_index] += 1
    assert(np.sum(x) == 13)
    return x


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

