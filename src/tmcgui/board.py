import random
from tmcgui.card import Card
from tmcgui.bid import Bid, BidButton, BidButtonSuit, SpecialBid
import copy

class Board:
    """
    Class for handling board.
    """

    # Class attributes
    deck = ["".join([suit, str(honour)]) for suit in "CDHS" for honour in range(2, 15)]
    bids = ["1C", "1D", "1H", "1S", "1N",
            "2C", "2D", "2H", "2S", "2N",
            "3C", "3D", "3H", "3S", "3N",
            "4C", "4D", "4H", "4S", "4N",
            "5C", "5D", "5H", "5S", "5N",
            "6C", "6D", "6H", "6S", "6N",
            "7C", "7D", "7H", "7S", "7N",
            ]

    def __init__(self, board_id):
        self.id = board_id
        self.players = [None, None, None, None]
        self.hands = [None, None, None, None]
        self.north = None
        self.south = None
        self.west = None
        self.east = None
        self.dealer = None
        self.turn = None
        self.history = []
        self.status = "bidding"
        self.bidding = []
        self.auction = []
        self.winning_bid = None
        self.winning_side = []
        self.vulnerable = [False, False, False, False]
        self.vulnerable_txt = ""
        self.available_bids = None
        self.special_bids = None
        self.declarer = None
        self.first_call_suit = {"C": {"0": None, "1": None},
                                "D": {"0": None, "1": None},
                                "H": {"0": None, "1": None},
                                "S": {"0": None, "1": None},
                                "N": {"0": None, "1": None}}
        self.dummy = None
        self.dummy_cards = None
        self.dummy_visible = False
        self.trump = None
        self.lead = None
        self.first_lead = True
        self.color_lead = None
        self.trick = [None, None, None, None]
        self.trick_db = [None, None, None, None]
        self.tricks = [0, 0]
        self.score = 0
        self.result = None
        #if board_id > 0:
            #self.shuffle()
        self.set_vulnerable()
        self.set_dealer()
        #else:
        self.south =  [Card('S14')] * 13
        self.west =  [Card('H14')] * 13
        self.north =  [Card('D14')] * 13
        self.east =  [Card('C14')] * 13

        self.passed_out = False

    def __repr__(self):
        return f"Board nr {self.id}"

    def shuffle(self):
        """
        Shuffling hands during initializing class instance. Marking last cards in hands.
        :return: None
        """
        shuffle_deck = self.deck
        random.seed(42)
        random.shuffle(shuffle_deck)
        self.north = [Card(symbol) for symbol in sorted(shuffle_deck[:13], key=lambda x: (x[0], -int(x[1:])))]
        self.south = [Card(symbol) for symbol in sorted(shuffle_deck[13:26], key=lambda x: (x[0], -int(x[1:])))]
        self.west = [Card(symbol) for symbol in sorted(shuffle_deck[26:39], key=lambda x: (x[0], -int(x[1:])))]
        self.east = [Card(symbol) for symbol in sorted(shuffle_deck[39:52], key=lambda x: (x[0], -int(x[1:])))]
        self.north[-1].last_card = True
        self.south[-1].last_card = True
        self.west[-1].last_card = True
        self.east[-1].last_card = True
        # Keep a copy of the entire hand
        self.hands = [copy.deepcopy(self.north), copy.deepcopy(self.east), copy.deepcopy(self.south), copy.deepcopy(self.west)]

    def set_vulnerable(self):
        if self.id < 1:
            return
        """
        Setting board condition according to board ID.
        :return: None
        """
        if self.id % 16 == 2 or self.id % 16 == 5 or self.id % 16 == 12 or self.id % 16 == 15:
            self.vulnerable = [True, False, True, False]
            self.vulnerable_txt = "NS"
        elif self.id % 16 == 4 or self.id % 16 == 7 or self.id % 16 == 10 or self.id % 16 == 13:
            self.vulnerable = [True, True, True, True]
            self.vulnerable_txt = "All"
        elif self.id % 16 == 3 or self.id % 16 == 6 or self.id % 16 == 9 or self.id % 16 == 0:
            self.vulnerable = [False, True, False, True]
            self.vulnerable_txt = "EW"
        else:
            self.vulnerable = [False, False, False, False]
            self.vulnerable_txt = "None"

    def set_dealer(self):
        """
        Setting dealer and turn to specific player according to board ID during initializing class instance.
        :return: None
        """
        if self.id % 4 == 1:
            self.dealer = 2
        elif self.id % 4 == 2:
            self.dealer = 3
        elif self.id % 4 == 3:
            self.dealer = 0
        else:
            self.dealer = 1
        self.turn = self.dealer
        self.auction = ['PAD_START'] * (self.id -1)
        # Getting starting bids
        self.available_bids, self.special_bids = self.get_available_bids()

    def draw_hand(self, win, hand, seat, user=False):
        """
        Dynamic drawing cards in the hand.
        :param win: pygame Surface instance
        :param hand: int
        :param seat: int
        :param user: boolean
        :return: None
        """

        # Assignation the hand to draw
        if hand == 0:
            drawing_hand = self.south
        elif hand == 1:
            drawing_hand = self.west
        elif hand == 2:
            drawing_hand = self.north
        else:
            drawing_hand = self.east

        # Entire size of the hand
        width = (len(drawing_hand) - 1) * 30 + 100
        height = (len(drawing_hand) - 1) * 30 + 100
        vertical = False

        # Coordinates for drawing according to taken seat
        if seat == 0:
            x = round(win.get_width() / 2 - width / 2)
            y = 790
        elif seat == 1:
            x = 60
            y = round(win.get_height() / 2 - height / 2)
            vertical = True
        elif seat == 2:
            x = round(win.get_width() / 2 - width / 2)
            y = 150
        else:
            x = 1000
            y = round(win.get_height() / 2 - height / 2)
            vertical = True

        # Iterating over cards in hand and drawing them sequentially with slight shift
        for card in drawing_hand:
            card.draw(win, x, y, user)
            if vertical:
                y += 30
            else:
                x += 30

    def set_actual_declarer(self, user):
        """
        Setting the declarer in current stage of bidding.
        :param user: int
        :return: None
        """
        if user in [0, 2]:
            self.first_call_suit[self.trump]["0"] = user
            self.declarer = (user, [0, 2], self.trump)
        else:
            self.first_call_suit[self.trump]["1"] = user
            self.declarer = (user, [1, 3], self.trump)

    def set_lead(self):
        """
        Setting the lead to player next to the declarer.
        :return: None
        """
        self.lead = self.declarer[0] + 1
        if self.lead > 3:
            self.lead = 0

    def make_bid(self, user, bid):
        """
        Making a call with specific bid. Updating turn and bidding history.
        :param user: int
        :param bid: string
        :return: None
        """
        if bid != "PASS":
            if bid == "X" or bid == "XX":
                self.winning_bid = self.winning_bid + "X"
            else:
                self.winning_bid = bid
                # Setting trump suit
                self.trump = self.winning_bid[1]
                if user in [0, 2]:
                    # Calling first time specific suit by NS side
                    if self.first_call_suit.get(self.trump).get("0") is None:
                        self.set_actual_declarer(user)
                    # Taking player, who first bid this suit, as declarer
                    else:
                        self.set_actual_declarer(self.first_call_suit.get(self.trump).get("0"))
                else:
                    # Calling first time specific suit by EW side
                    if self.first_call_suit.get(self.trump).get("1") is None:
                        self.set_actual_declarer(user)
                    # Taking player, who first bid this suit, as declarer
                    else:
                        self.set_actual_declarer(self.first_call_suit.get(self.trump).get("1"))

            if user in [0, 2]:
                self.winning_side = [0, 2]
            else:
                self.winning_side = [1, 3]
        # Updating turn
        self.turn += 1
        if self.turn > 3:
            self.turn = 0
        # Updating bidding history and getting available bids for next turn
        self.bidding.append(Bid(bid))
        self.auction.append(bid)
        self.available_bids, self.special_bids = self.get_available_bids()
        # Checking that bidding phase is over
        if self.end_bidding():
            self.available_bids = None
            self.special_bids = None

    def get_available_bids(self):
        """
        Getting available bids for current turn.
        :return: tuple
        """
        special_bids = ["PASS"]
        if self.winning_bid:
            indx = self.bids.index(self.winning_bid[:2])
            if self.turn not in self.winning_side:
                # Adding redouble to special bids, when opponents called double
                if self.winning_bid[-1] == "X" and len(self.winning_bid) == 3:
                    special_bids = special_bids + ["XX"]
                elif self.winning_bid[-1] == "X" and len(self.winning_bid) == 4:
                    pass
                # Adding double to special bids, when opponents called any bid (exclude double)
                else:
                    special_bids = special_bids + ["X"]
            # Getting higher bids than already has been bid
            available_bids = self.bids[indx + 1:]
        else:
            # No one bids, getting all bids
            available_bids = self.bids
        # Creating dictionary of available bids, level bids are keys, list of appropriate denomination bids are values
        available_bids_dictio = dict()
        for b in available_bids:
            if available_bids_dictio.get(b[0]):
                available_bids_dictio[b[0]].append(b[1])
            else:
                available_bids_dictio[b[0]] = [b[1]]
        # List of special bids with initialized instances of SpecialBid class
        special_bids = [SpecialBid(b) for b in special_bids]
        # List of availabl bids with initialized instances of BidButton and BidButtonSuit classes
        normal_bids = dict()
        for k, values in available_bids_dictio.items():
            new_key = BidButton(k)
            normal_bids[new_key] = [BidButtonSuit(v, new_key) for v in values]
        return normal_bids, special_bids

    def end_bidding(self):
        """
        Checking that bidding phase is over.
        :return: boolean
        """
        bidding = [x for x in self.bidding if x is not None]
        
        # Auction begins with four consecutive passes
        if  len(bidding) == 4:
            if all(b.bid == "PASS" for b in bidding[:4]):
                self.status = "play"
                self.passed_out = True
                return True
        # Three consecutive passes following a bid, double or redouble
        if len(bidding) > 3:
            if all(b.bid == "PASS" for b in bidding[-3:]) and len(bidding) > 3:
                self.status = "play"
                # Setting the lead, the trump and the dummy
                self.set_lead()
                for player in self.declarer[1]:
                    if self.declarer[0] != player:
                        self.dummy = player
                self.turn = self.lead
                if self.trump != "N":
                    self.setting_trumps()
                return True
        return False

    def setting_trumps(self):
        """
        Marking cards, which have the same suit as trump
        :return: None
        """
        for hand in [self.south, self.north, self.east, self.west]:
            for card in hand:
                if card.symbol[0] == self.trump:
                    card.trump = True

    def make_move(self, card_symbol):
        """
        Adding the played card to the trick. Removing this card from the hand.
        Assignation who takes trick, when all players played card.
        Updating the turn, the lead, tricks history and amount of tricks each side.
        :param card_symbol: string
        :return: None
        """
        self.first_lead = False
        # All players added card to trick, reset trick and appending tricks history
        if all(t for t in self.trick):
            self.history.append(self.trick_db)
            self.trick = [None, None, None, None]
            self.trick_db = [None, None, None, None]
        # Assignation current hand
        card = None
        if self.turn == 0:
            hand = self.south
        elif self.turn == 1:
            hand = self.west
        elif self.turn == 2:
            hand = self.north
        else:
            hand = self.east
        # Removing the played card from the hand
        for c in hand:
            if c.symbol == card_symbol:
                card = c
                hand.remove(c)
        
        assert card is not None, f"card is None: hand={hand}, c={card_symbol}"
        
        if len(hand) > 0:
            hand[-1].last_card = True
        # Card is visible for everyone
        card.hidden = False
        # Adding card to the trick
        self.trick[self.turn] = card
        self.trick_db[self.turn] = card.symbol
        # Setting the suit that others must play if able to do so
        if self.lead is not None:
            self.color_lead = card_symbol[0]
        self.lead = None
        # Setting dummy's hand as visible after first lead
        if not self.dummy_visible:
            self.dummy_visible = True
            if self.dummy == 0:
                self.dummy_cards = self.south
            elif self.dummy == 1:
                self.dummy_cards = self.west
            elif self.dummy == 2:
                self.dummy_cards = self.north
            else:
                self.dummy_cards = self.east
            for c in self.dummy_cards:
                c.hidden = False
        # Updating turn
        self.turn += 1
        if self.turn > 3:
            self.turn = 0
        # Four cards in the trick
        if all(t for t in self.trick):
            # Updating values of cards in trick
            for t in self.trick:
                t.set_value(self.color_lead)
            # Who collects the trick
            self.turn = self.trick.index(max(self.trick))
            self.trick_db[self.turn] = self.trick_db[self.turn] + "*"
            if self.turn in [0, 2]:
                self.tricks[0] += 1
            else:
                self.tricks[1] += 1
            self.lead = self.turn
            self.color_lead = None
            # Ending the board, when any card hasn't left
            if len(hand) == 0:
                self.set_score()
                self.status = "score"

    def set_score(self):
        """
        Calculating final score depending on the board condition, tricks result and played contract.
        :return: None
        """
        # Tricks that should be taken
        level_contract = int(self.winning_bid[0]) + 6
        # Tricks taken
        if self.declarer[1] == [0, 2]:
            vul = self.vulnerable[0]
            taken_tricks = self.tricks[0]
        else:
            vul = self.vulnerable[1]
            taken_tricks = self.tricks[1]
        # Final result
        score = taken_tricks - level_contract
        # Updating attribute for displaying final result
        if score == 0:
            self.result = self.winning_bid + "=="
        elif score > 0:
            self.result = self.winning_bid + f"+{score}"
        else:
            self.result = self.winning_bid + f"{score}"
        doubled = False
        redoubled = False
        # Determine if game was doubled or redoubled
        if self.winning_bid.endswith("X"):
            if len(self.winning_bid) == 3:
                doubled = True
            else:
                redoubled = True
        making_game = False
        making_slam = False
        making_grand_slam = False
        # Eventual bonus for game, slam and grand slam
        if level_contract == 13 and score == 0:
            making_grand_slam = True
        elif level_contract == 12 and score >= 0:
            making_slam = True
        if score >= 0:
            if (self.trump == "C" or self.trump == "D") and level_contract >= 11:
                making_game = True
            elif (self.trump == "H" or self.trump == "S") and level_contract >= 10:
                making_game = True
            elif self.trump == "N" and level_contract >= 9:
                making_game = True
            # Adding bonus points for game, slam or grand slam to the score
            if vul:
                if making_grand_slam:
                    self.score += 1500
                elif making_slam:
                    self.score += 750
                elif making_game:
                    self.score += 500
                else:
                    self.score += 50 + (doubled * 50 + redoubled * 150)
            else:
                if making_grand_slam:
                    self.score += 1000
                elif making_slam:
                    self.score += 500
                elif making_game:
                    self.score += 300
                else:
                    self.score += 50 + (doubled * 50 + redoubled * 150)
            # Adding points for tricks and overtricks to the score
            if vul:
                if self.trump == "C" or self.trump == "D":
                    self.score += (level_contract - 6) * 20 + score * (20 + 180 * doubled + 380 * redoubled)
                elif self.trump == "H" or self.trump == "S":
                    self.score += (level_contract - 6) * 30 + score * (30 + 170 * doubled + 370 * redoubled)
                elif self.trump == "N":
                    self.score += 40 + (level_contract - 7) * 30 + score * (30 + 170 * doubled + 370 * redoubled)
            else:
                if self.trump == "C" or self.trump == "D":
                    self.score += (level_contract - 6) * 20 + score * (20 + 80 * doubled + 180 * redoubled)
                elif self.trump == "H" or self.trump == "S":
                    self.score += (level_contract - 6) * 30 + score * (30 + 70 * doubled + 170 * redoubled)
                elif self.trump == "N":
                    self.score += 40 + (level_contract - 7) * 30 + score * (30 + 70 * doubled + 170 * redoubled)
        # Subtracting points for 1 undertrick
        elif score == -1:
            if vul:
                self.score -= 100 + 100 * doubled + 300 * redoubled
            else:
                self.score -= 50 + 50 * doubled + 150 * redoubled
        # Subtracting points for 2 undertricks
        elif score == -2:
            if vul:
                self.score -= 200 + 300 * doubled + 800 * redoubled
            else:
                self.score -= 100 + 200 * doubled + 500 * redoubled
        # Subtracting points for 3 undertricks
        elif score == -3:
            if vul:
                self.score -= 300 + 500 * doubled + 1300 * redoubled
            else:
                self.score -= 150 + 350 * doubled + 850 * redoubled
        # Subtracting points for 4 undertricks or more
        else:
            if vul:
                self.score -= 300 + 500 * doubled + 1300 * redoubled + abs(score + 3) * (100 + 200 * doubled + 500 * redoubled)
            else:
                self.score -= 150 + 350 * doubled + 850 * redoubled + abs(score + 3) * (50 + 250 * doubled + 550 * redoubled)
