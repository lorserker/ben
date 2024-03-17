import sys
import re
from collections import deque
from typing import NamedTuple

class Deal(NamedTuple):
    dealer: str
    vulnerable: str
    hands: str

import numpy as np
# Constants for suits
C_SPADES = 3
C_HEARTS = 2
C_DIAMONDS = 1
C_CLUBS = 0
C_NT = 4
C_NORTH = 0
C_EAST = 1
C_SOUTH = 2
C_WEST = 3
C_NONE = 0
C_WE = 1
C_NS = 2
C_BOTH = 3

board = np.zeros((4, 4), dtype=int)

board[C_NORTH, C_NONE] = 1
board[C_EAST, C_NS] = 2
board[C_SOUTH, C_WE] = 3
board[C_WEST, C_BOTH] = 4
board[C_NORTH, C_NS] = 5
board[C_EAST, C_WE] = 6
board[C_SOUTH, C_BOTH] = 7
board[C_WEST, C_NONE] = 8
board[C_NORTH, C_WE] = 9
board[C_EAST, C_BOTH] = 10
board[C_SOUTH, C_NONE] = 11
board[C_WEST, C_NS] = 12
board[C_NORTH, C_BOTH] = 13
board[C_EAST, C_NONE] = 14
board[C_SOUTH, C_NS] = 15
board[C_WEST, C_WE] = 16


class TypeHand:
    def __init__(self):
        self.suit = [""] * 4


def print_hand(hand):
    for i in range(4):
        for j in range(4):
            print(hand[i].suit[3 - j], end="")
            if j == 3:  # Checks if we've reached the last suit
                print("", end=" ")  # Print space at the end of each player's hand
            else:
                print("", end=".")
    print()


def encode_board(hand, dealer, vulnerability, deal):
    board_extension = ((deal - 1) // 16) % 16
    str_Deal = format(board_extension, 'x') + format(dealer * 4 + vulnerability, 'x')
    encryption_byte = board[dealer,vulnerability]

    for j in range(1, 14):
        sum_value = 0
        str_cards = "AKQJT98765432"[j - 1]
        for i in range(4):
            for player_index in range(4):
                if str_cards in hand[player_index].suit[i]:
                    sum_value += int(player_index * (4 ** i))
        # ---coding
        sum_value = encryption_byte ^ sum_value
        str_cards = format(sum_value, 'x')
        if len(str_cards) == 1:
            str_cards = "0" + str_cards
        str_Deal += str_cards.upper()

    return str_Deal.upper()  # Convert the entire string to uppercase for consistency with VBA output


def decode_board(encoded_str_Deal):
    # Initialize the hand
    hand = [TypeHand() for _ in range(4)]

    board_extension = int(encoded_str_Deal[0], 16)
    number = int(encoded_str_Deal[1], 16)
    dealer = number // 4
    vulnerable = number % 4
    deal = board_extension * 16 + board[dealer][vulnerable]
    encryption_byte = board[dealer][vulnerable]

    card_index = 2
    for j in range(1, 14):
        str_card = "AKQJT98765432"[j - 1]
        str_number = encoded_str_Deal[card_index:card_index + 2]
        number = int(str_number, 16)
        number = encryption_byte ^ number
        lbloki = [0] * 4
        lbloki[0] = number % 4
        lbloki[1] = (number // 4) % 4
        lbloki[2] = (number // 16) % 4
        lbloki[3] = number // 64
        card_index += 2

        for i in range(4):
            k = lbloki[i]
            hand[k].suit[i] += str_card

    return hand, dealer, vulnerable, deal

def transform_hand(hands):
    #print(hands)
    hand = [TypeHand() for _ in range(4)]

    for i in range(4):
        suits = hands[i].split('.')
        hand[i].suit[C_CLUBS] = suits[3]
        hand[i].suit[C_DIAMONDS] = suits[2]
        hand[i].suit[C_HEARTS] = suits[1]
        hand[i].suit[C_SPADES] = suits[0]
    
    return hand

def load(fin):
    boards = set()
    auction_lines = []
    inside_auction_section = False
    dealer, vulnerable = None, None
    dealnumber = 0
    try:
        for line in fin:
            if line.startswith("% PBN") or line == "\n":
                if dealer != None:
                    dealnumber += 1
                    encoded_str_deal = encode_board(transform_hand(hands_nesw), dealer, vulnerable, dealnumber)
                    # Do we have the board all ready, then discard it with a message
                    if encoded_str_deal in boards:
                        print("Repeated",hands_nesw)
                    else:
                        boards.add(encoded_str_deal)      
                    auction_lines = []
                    dealer= None
            if line.startswith('[Dealer'):
                dealer_str = extract_value(line)
                dealer = {'N': C_NORTH, 'E': C_EAST, 'S': C_SOUTH, 'W': C_WEST}.get(dealer_str, C_NORTH)
            if line.startswith('[Vulnerable'):
                vuln_str = extract_value(line)
                vulnerable = {'None': C_NONE, 'NS': C_NS, 'EW': C_WE, 'All': C_BOTH}.get(vuln_str, C_NONE)
            if line.startswith('[Deal '):
                hands_pbn = extract_value(line)
                [seat, hands] = hands_pbn.split(':')
                hands_nesw = [''] * 4
                first_i = 'NESW'.index(seat)
                for hand_i, hand in enumerate(hands.split()):
                    hands_nesw[(first_i + hand_i) % 4] = hand
            if line.startswith('[Auction'):
                inside_auction_section = True
                continue  
            if inside_auction_section:
                if line.startswith('[') or line == "\n":  # Check if it's the start of the next tag
                    inside_auction_section = False
                else:
                    # Convert bids
                    line = line.strip().replace('.','').replace("NT","N").replace("Pass","P").replace("Double","X").replace("Redouble","XX").replace('AP','P P P')
                    # Remove extra spaces
                    line = re.sub(r'\s+', ' ', line)
                    # Remove alerts
                    line = re.sub(r'=\d{1,2}=', '', line)
                    auction_lines.append(line)  

            else:
                continue
    except:
        print(line)
    # Handling last deal if ny
    if dealer != None:
        encoded_str_deal = encode_board(transform_hand(hands_nesw), dealer, vulnerable, dealnumber)
        boards.append(encoded_str_deal)      
    return boards

def extract_value(s: str) -> str:
    return s[s.index('"') + 1 : s.rindex('"')]

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python pbn2bba.py input_file.pbn")
        sys.exit(1)
    
    input_file = sys.argv[1]
    try:
        with open(input_file, "r", encoding='utf-8') as file:  # Open the input file with UTF-8 encoding
            lines = file.readlines()
        
        with open('input.bba', 'w', encoding='utf-8') as file:  # Open the output file with UTF-8 encoding
            boards = load(lines)
            for board in boards:
                file.write(board + "\n")

            print("File input.bba generated")
    except Exception as ex:
        print('Error:', ex)
        raise ex
