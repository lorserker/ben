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



def decode_board(encoded_str_deal):
    # Initialize the hand
    hand = [TypeHand() for _ in range(4)]

    board_extension = int(encoded_str_deal[0], 16)
    number = int(encoded_str_deal[1], 16)
    dealer_i = number // 4
    vulnerable = number % 4
    deal_no = board_extension * 16 + board[dealer_i][vulnerable]
    encryption_byte = board[dealer_i][vulnerable]

    card_index = 2
    for j in range(1, 14):
        str_card = "AKQJT98765432"[j - 1]
        str_number = encoded_str_deal[card_index:card_index + 2]
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
    dealer = "NESW"[dealer_i]
    vulnerable = ['None', 'E-W', 'N-S', 'Both'][vulnerable]
    return hand, dealer, vulnerable, deal_no


hand = [TypeHand() for _ in range(4)]

# Assigning values to hand suits
hand[0].suit[C_CLUBS] = "987"
hand[0].suit[C_DIAMONDS] = "KQJT7"
hand[0].suit[C_HEARTS] = "A74"
hand[0].suit[C_SPADES] = "A6"

hand[1].suit[C_CLUBS] = "T652"
hand[1].suit[C_DIAMONDS] = "3"
hand[1].suit[C_HEARTS] = "QJ82"
hand[1].suit[C_SPADES] = "J732"

hand[2].suit[C_CLUBS] = "J3"
hand[2].suit[C_DIAMONDS] = "A9842"
hand[2].suit[C_HEARTS] = "K3"
hand[2].suit[C_SPADES] = "T984"

hand[3].suit[C_CLUBS] = "AKQ4"
hand[3].suit[C_DIAMONDS] = "65"
hand[3].suit[C_HEARTS] = "T965"
hand[3].suit[C_SPADES] = "KQ5"

dealnumber = 108
print(dealnumber, "West", "NS")
print_hand(hand)
encoded_str_Deal = encode_board(hand, C_WEST, C_NS, dealnumber)
print(encoded_str_Deal)
decoded_hand, dealer, vulnerability, dealnumber = decode_board(encoded_str_Deal)
print(dealnumber, dealer, vulnerability)
print_hand(decoded_hand)
