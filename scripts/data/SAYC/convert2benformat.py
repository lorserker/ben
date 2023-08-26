import tarfile
from pathlib import Path
import os
import json

# Vulnerability translation
vulnerabilityString = {
    0: 'E-W',
    1: 'None',
    2: 'N-S',
    3: 'E-W',
    4: 'Both',
    5: 'N-S',
    6: 'E-W',
    7: 'Both',
    8: 'None',
    9: 'E-W',
    10: 'Both',
    11: 'None',
    12: 'N-S',
    13: 'Both',
    14: 'None',
    15: 'N-S',
}

# Dealer translation
dealerString = {
    0: 'W',
    1: 'N',
    2: 'E',
    3: 'S',
}

class Suit:
    def __init__(self, name):
        if name not in 'CDHS':
            raise ValueError(f"Unknown Suit {name}")
        self.name = name

    SUITS = ['C', 'D', 'H', 'S']

    @classmethod
    def from_char(cls, suit_char):
        return cls(suit_char)


class Rank:
    def __init__(self, name):
        self.name = name

    def display_rank(self):
        return self.name

    def index(self):
        return '23456789TJQKA'.index(self.name)

    def high_card_points(self):
        return {
            'A': 4,
            'K': 3,
            'Q': 2,
            'J': 1,
        }.get(self.name, 0)

    @classmethod
    def pbn_rank_from_display_rank(cls, display_rank):
        if display_rank == '10':
            return 'T'
        return display_rank

    _cached_ranks = {}

    @classmethod
    def from_char(cls, rank_char):
        rank = cls._cached_ranks.get(rank_char)
        if not rank:
            rank = cls(rank_char)
            cls._cached_ranks[rank_char] = rank
        return rank


class Hand:
    def __init__(self, cards=None):
        self.cards = cards if cards is not None else []


class Card:
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    def short_name(self):
        return self.rank.name + self.suit.name

    def high_card_points(self):
        return self.rank.high_card_points()

    def identifier(self):
        suit_index = 'CDHS'.index(self.suit.name)
        return suit_index * 13 + self.rank.index()

    @classmethod
    def from_identifier(cls, identifier):
        suit_index = identifier // 13
        card_index = identifier - suit_index * 13
        suit_char = 'CDHS'[suit_index]
        rank_char = '23456789TJQKA'[card_index]
        return cls.from_short_name(rank_char + suit_char)

    _cached_cards = {}

    @classmethod
    def from_short_name(cls, short_name):
        card = cls._cached_cards.get(short_name)
        if not card:
            card = cls(Suit.from_char(short_name[1]), Rank.from_char(short_name[0]))
            cls._cached_cards[short_name] = card
        return card


class Deal:
    def __init__(self, hands):
        self.hands = hands

    @classmethod
    def from_identifier(cls, identifier_string):
        hands = Deal._empty_hands()
        hex_chars = '0123456789abcdef'
        for char_index, hex_char in enumerate(identifier_string):
            hex_index = hex_chars.index(hex_char)
            high_hand_index = hex_index // 4
            low_hand_index = hex_index - high_hand_index * 4
            high_card = Card.from_identifier(char_index * 2 + 0)
            low_card = Card.from_identifier(char_index * 2 + 1)
            hands[high_hand_index].cards.append(high_card)
            hands[low_hand_index].cards.append(low_card)
        return cls(hands)

    @staticmethod
    def _empty_hands():
        return [Hand() for _ in range(4)]


# Example call to Deal.from_identifier
#identifier_param = "d2b678746acce6548e00609ff7"
#deal_from_identifier = Deal.from_identifier(identifier_param)


with tarfile.open('bidder-data.tar.xz', 'r:xz') as tar:
    tar.extractall(path='.')

# Iterate through all files in the extracted directory
for root, dirs, files in os.walk('bidder-data'):
    for file_name in files:
        if file_name.endswith('.json'):
            # Process only JSON files
            file_path = os.path.join(root, file_name)
            
            # Open and read the JSON content
            with open(file_path, 'r') as json_file:
                json_content_array = json.load(json_file)
                
                for json_content in json_content_array:
                    # Extract "board" and "calls" values
                    board = json_content.get("board")
                    calls = json_content.get("calls")
                    # Split the board value
                    board_parts = board.split('-')
                    first_part = int(board_parts[0])
                        
                    # Get the vulnerability string and dealer based on translations
                    vulnerability = vulnerabilityString[first_part % 16]
                    dealer = dealerString[first_part % 4]

                    second_part = board_parts[1]
                    
                    # Call Deal.from_identifier with the second part
                    deal_from_identifier = Deal.from_identifier(second_part)
                    
                    # Initialize a list to store the formatted hands
                    formatted_hands = []

                    # Format and store each hand
                    for hand in deal_from_identifier.hands:
                        formatted_hand = []

                        # Group cards by suit and format them
                        for suit_name in ['S', 'H', 'D', 'C']:
                            suit_cards = [card.rank.display_rank() for card in hand.cards if card.suit.name == suit_name]
                            suit_cards_str = ''.join(reversed(suit_cards)) if suit_cards else ''
                            formatted_hand.append(suit_cards_str)

                        formatted_hands.append('.'.join(formatted_hand))

                    # Join the formatted hands with a space delimiter
                    hands_str = ' '.join(formatted_hands)

                    #print("Board:", board)
                    print(hands_str)
                    print(dealer, vulnerability, ' '.join(calls))

#print("Processing of all files completed.")