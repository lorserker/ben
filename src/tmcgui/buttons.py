import pygame


pygame.init()


class Button:
    """
    Class for handling buttons.
    """

    # Class attributes
    button_font = pygame.font.SysFont("Arial", 32)
    button_font2 = pygame.font.SysFont("Arial", 20)

    def __init__(self, width, height, color, string, x, y):
        self.width = width
        self.height = height
        self.color = color
        self.string = string
        self.x = x
        self.y = y

    def draw(self, win):
        """
        Drawing the button and text inside it.
        :param win: pygame Surface instance
        :return: None
        """
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.height))
        button_text = self.button_font.render("{}".format(self.string), True, (255, 255, 255))
        win.blit(button_text, (round(self.x + self.width / 2 - button_text.get_width() / 2), round(self.y + self.height / 2 - button_text.get_height() / 2)))

    def on_button(self):
        """
        Checking that button is  clicked or not.
        :return: boolean
        """
        pos = pygame.mouse.get_pos()
        if self.x <= pos[0] <= self.x + self.width:
            if self.y <= pos[1] <= self.y + self.height:
                return True
        return False


