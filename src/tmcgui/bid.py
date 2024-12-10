import pygame


pygame.init()


class Bid:
    """
    Class for handling bids displayed in the bidding table.
    """

    def __init__(self, bid):
        self.bid = bid
        self.rect = None

    def __repr__(self):
        return f"{self.bid}"


class BidButton:
    """
    Class for handling level bids.
    """

    def __init__(self, bid):
        self.bid = bid
        self.active = False
        self.rect = None

    def click(self):
        """
        Checking that specific level bid is clicked.
        :return: boolean
        """
        if self.rect == None: return False
        pos = pygame.mouse.get_pos()
        if self.rect[0] < pos[0] < self.rect[0] + self.rect[2]:
            if self.rect[1] < pos[1] < self.rect[1] + self.rect[3]:
                return True
        return False

    def deactivate(self):
        """
        When the other level bid is clicked, setting 'active' attribute of this level bid to False.
        :return: None
        """
        self.active = False

    def __repr__(self):
        return f"Button {self.bid}"


class BidButtonSuit:
    """
    Class for handling denomination bids.
    """

    def __init__(self, bid, first_button):
        self.bid = bid
        self.first_part_bid = first_button
        self.rect = None
        self.bidded = False

    def click(self):
        """
        Checking that the denomination bid is clicked.
        :return: boolean
        """
        if self.rect == None: return False
        pos = pygame.mouse.get_pos()
        if self.rect[0] < pos[0] < self.rect[0] + self.rect[2]:
            if self.rect[1] < pos[1] < self.rect[1] + self.rect[3]:
                self.bidded = True
                return True
        return False

    def __repr__(self):
        return f"ButtonSuit {self.bid}"


class SpecialBid:
    """
    Class for handling special bids like fold and double/redouble.
    """

    def __init__(self, bid):
        self.bid = bid
        self.bidded = False
        self.rect = None
        if bid == "X":
            self.text = 'X'
        elif bid == "XX":
            self.text = 'XX'
        else:
            self.text = 'P'

    def click(self):
        """
        Checking that the special bid is clicked.
        :return: boolean
        """
        if self.rect == None: return False
        pos = pygame.mouse.get_pos()
        if self.rect[0] < pos[0] < self.rect[0] + self.rect[2]:
            if self.rect[1] < pos[1] < self.rect[1] + self.rect[3]:
                self.bidded = True
                return True
        return False

    def __repr__(self):
        return f"SpecialButton {self.text}"
