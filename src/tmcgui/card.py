import pygame
import os
from pathlib import Path


base_directory = Path(__file__).parent
pygame.init()


# Images of cards
# Create a list of back colors
back_colors = ["blue", "gray", "green", "purple", "red", "yellow"]

# Create a dictionary to store the loaded back images
back_images_loaded = {}

# Load the back images for all back colors
for color in back_colors:
    file_name = f"{color}_back.png"
    back_images_loaded[color] = pygame.image.load(os.path.join(base_directory, f"images/deck/width 100/{file_name}"))

# Create a list of suits and values
suits = ["C", "D", "H", "S"]
values = list(range(2, 15))  # 2-14 (inclusive)

# Create a dictionary to store the loaded card images
card_images_loaded = {}

# Load the card images for all combinations of suits and values
for suit in suits:
    for value in values:
        card_name = f"{suit}{value}"
        file_name = f"{card_name}.png"
        card_images_loaded[card_name] = pygame.image.load(os.path.join(base_directory, f"images/deck/width 100/{file_name}"))


class Card:
    """
    Class for handling single card.
    """

    def __init__(self, symbol):
        self.symbol = symbol
        self.trump = False
        self.value = int(symbol[1:]) + 100 * self.trump
        self.hidden = True
        self.last_card = False
        self.suited_with_lead = False
        self.rect = None

    def __lt__(self, other):
        """
        Comparing two different cards based on their value.
        :param other: Card instance
        :return: boolean
        """
        if self.value < other.value:
            return True
        return False
        
    def get_ben_value(self):
        return self.symbol.replace("14","A").replace("13","K").replace("12","Q").replace("11","J").replace("10","T")

    def set_value(self, lead_color):
        """
        Updating value of card depends on the suit of card and the suit of the lead.
        :param lead_color: string
        :return: None
        """
        if self.symbol[0] != lead_color and not self.trump:
            self.value = 0
        else:
            self.value = int(self.symbol[1:]) + 100 * self.trump

    def draw(self, win, x, y, user):
        """
        Drawing card and setting up rect attribute.
        :param win: pygame Surface instance
        :param x: int
        :param y: int
        :param user: boolean
        :return: None
        """
        if user:
            win.blit(card_images_loaded[self.symbol], (x, y))
        else:
            if self.hidden:
                win.blit(back_images_loaded["blue"], (x, y))
            else:
                win.blit(card_images_loaded[self.symbol], (x, y))
        if self.last_card:
            self.rect = (x, y, 100, 153)
        else:
            self.rect = (x, y, 30, 153)

    def click(self):
        """
        Checking that card is clicked or not
        :return: boolean
        """
        pos = pygame.mouse.get_pos()
        if self.rect[0] < pos[0] < self.rect[0] + self.rect[2]:
            if self.rect[1] < pos[1] < self.rect[1] + self.rect[3]:
                return True
        return False

    def __repr__(self):
        return f"{self.symbol}"

    def __str__(self):
        return f"{self.symbol}"