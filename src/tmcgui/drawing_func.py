import pygame
import os
from pathlib import Path


base_directory = Path(__file__).parent

def load_bidding_image(suit):
    image = pygame.image.load(os.path.join(base_directory, "images/bid/"+suit+".png"))
    original_image_rect = image.get_rect()
    # Scale the image to 80% of its original size
    scale_factor = 0.8
    new_image_size = (int(original_image_rect.width * scale_factor), int(original_image_rect.height * scale_factor))
    image = pygame.transform.scale(image, new_image_size)
    return image


# Images of cards
bid_font = pygame.font.SysFont("Arial", 24)
C = load_bidding_image("clubs")
D = load_bidding_image("diamonds")
H = load_bidding_image("hearts")
S = load_bidding_image("spades")
N = bid_font.render("NT", 1, (0, 0, 0))


def redraw_sitting(win, font, table, user):
    """
    Drawing seats on the table.
    :param win: pygame Surface instance
    :param font: pygame Surface instance
    :param table: Table instance
    :param user: Player instance
    :return: None
    """
    # Background
    win.fill((40, 125, 67))
    # Iterating over players at the table
    player_text = None
    for i, p in enumerate(table.players):
        # Seat taken
        if p:
            if i == 0:
                player_text = font.render(f"S - {p}", 1, (0, 0, 0))
            if i == 1:
                player_text = font.render(f"W - {p}", 1, (0, 0, 0))
            if i == 2:
                player_text = font.render(f"N - {p}", 1, (0, 0, 0))
            if i == 3:
                player_text = font.render(f"E - {p}", 1, (0, 0, 0))
        # Free seat
        else:
            if i == 0:
                player_text = font.render("S", 1, (0, 0, 0))
            elif i == 1:
                player_text = font.render("W", 1, (0, 0, 0))
            elif i == 2:
                player_text = font.render("N", 1, (0, 0, 0))
            else:
                player_text = font.render("E", 1, (0, 0, 0))
        # Rotating players for display client username on the bottom of window
        i -= user.position
        if i < 0:
            i += 4
        if i == 0:
            win.blit(player_text, (round(win.get_width() / 2 - player_text.get_width() / 2), 950))
        elif i == 1:
            win.blit(player_text, (round(120 - player_text.get_width() / 2), 100))
        elif i == 2:
            win.blit(player_text, (round(win.get_width() / 2 - player_text.get_width() / 2), 100))
        elif i == 3:
            win.blit(player_text, (round(1050 - player_text.get_width() / 2), 100))


def draw_cards(win, font, table, board, user):
    """
    Drawing all cards in hands.
    :param win: pygame Surface instance
    :param font: pygame Surface instance
    :param table: Table instance
    :param board: Board instance
    :param user: Player instance
    :return: None
    """
    
    dealer = ""
    # Assignation the dealer text
    if board.dealer == 0:
        dealer = "Dealer: S"
    elif board.dealer == 1:
        dealer = "Dealer: W"
    elif board.dealer == 2:
        dealer = "Dealer: N"
    elif board.dealer == 3:
        dealer = "Dealer: E"

    # Drawing dealer info
    dealer_text = font.render(dealer, 1, (0, 0, 0))
    win.blit(dealer_text, (20, 10))
    # Drawing board condition
    vulnerable_text = font.render(board.vulnerable_txt, 1, (0, 0, 0))
    win.blit(vulnerable_text, (20, 50))

    # Iterating over players at the table
    for i, p in enumerate(table.players):
        # Client can see only his cards
        is_my_hand = False
        if p == user.username:
            is_my_hand = True
        # Dummy can see declarer hand after first lead
        if board.dummy is not None and not board.first_lead:
            if user.position in board.declarer[1] and i in board.declarer[1]:
                is_my_hand = True
        # Rotating hands for drawing client cards always on the bottom of window
        hand = i
        i -= user.position
        if i < 0:
            i += 4
        # Drawing hand
        board.draw_hand(win, hand, i, is_my_hand)


def redraw_bidding(win, font, buttons, table, board, user, normal_bids, special_bids):
    """
    Drawing all elements during bidding phase.
    :param win: pygame Surface instance
    :param font: pygame Surface instance
    :param font2: pygame Surface instance
    :param buttons: list
    :param table: Table instance
    :param board: Board instance
    :param user: Player instance
    :param normal_bids: dictionary
    :param special_bids: list
    :return: None
    """
    # Drawing seats
    redraw_sitting(win, font, table, user)
    # Drawing board info
    if board.id > 0:
        board_text = font.render(f"Deal {board.id}", 1, (0, 0, 0))
        win.blit(board_text, (1000, 20))

    # Drawing cards
    draw_cards(win, font, table, board, user)
    # Drawing bidding box, when it's user turn
    if board.turn == user.position and normal_bids != None:
        last_x = 0
        for i, bids in enumerate(normal_bids.items()):
            # Drawing level bids sequentially
            x = 835 + i * 38
            y = 850
            rect = (x, y, 35, 35)
            bids[0].rect = rect
            bids[0].rect = (x, y, 35, 35)
            text = font.render(bids[0].bid, 1, (0, 0, 0))
            # Marking active the level bid by light green color
            if bids[0].active:
                pygame.draw.rect(win, (49, 224, 105), rect)
            else:
                pygame.draw.rect(win, (255, 255, 255), rect)
            win.blit(text, (round(x + rect[2] / 2 - text.get_width() / 2), round(y + rect[3] / 2 - text.get_height() / 2)))
            last_x = x
            # Drawing denomination bids for specific level bid
            if bids[0].active:
                for j, suitbid in enumerate(bids[1]):
                    image = eval(suitbid.bid)
                    x = 835 + j * 38
                    y = 890
                    rect = (x, y, 35, 35)
                    suitbid.rect = rect
                    pygame.draw.rect(win, (255, 255, 255), rect)
                    win.blit(image, (round(x + rect[2] / 2 - image.get_width() / 2), round(y + rect[3] / 2 - image.get_height() / 2)))
        # Drawing special bids like fold and double/redouble
        for j, b in enumerate(special_bids):
            if last_x:
                x = last_x + 38 + j * 45
            else:
                x = 835 + j * 45
            y = 850
            rect = (x, y, 42, 35)
            b.rect = rect
            pygame.draw.rect(win, (255, 255, 255), rect)
            text = font.render(b.text, 1, (0, 0, 0))
            win.blit(text, (round(x + rect[2] / 2 - text.get_width() / 2),
                            round(y + rect[3] / 2 - text.get_height() / 2)))

    # Drawing bidding table rotated for client
    seats_rotation = [(1, "W"), (2, "N"), (3, "E"), (0, "S")]
    for i, seat in enumerate(seats_rotation):
        # Red background/white text for vulnerable
        if board.vulnerable[seat[0]]:
            color_rect = (166, 20, 3)
            color_text = (255, 255, 255)
        # White background/black text for vulnerable
        else:
            color_rect = (255, 255, 255)
            color_text = (0, 0, 0)
        header_txt = font.render(seat[1], 1, color_text)
        pygame.draw.rect(win, color_rect, (round(win.get_width() / 2 + (i - 2) * 60), 310, 60, 38))
        win.blit(header_txt, (round(win.get_width() / 2 + (i - 1.5) * 60 - header_txt.get_width() / 2), round(310 + 19 - header_txt.get_height() / 2)))
    # Drawing called bids
    for i, b in enumerate(board.bidding):
        if b:
            row = ((board.dealer) + i) // 4
            # South is dealer)
            if (board.dealer == 0):
                row = (i+2) // 4
            if (board.dealer == 2):
                row = (i+1) // 4
            if (board.dealer == 3):
                row = (i+2) // 4
            y = 345 + (row) * 35 + 19
            if (b.bid == "X" or b.bid == "XX"):
                x = (((board.dealer - 1) + i) % 4 - 1.5) * 60
                text = font.render(b.bid, 1, (0, 50, 0))                 
                win.blit(text, (round(win.get_width() / 2 + x - text.get_width() / 2), round(y - text.get_height() / 2)))

            elif (b.bid[1] in {"C","D","H","S"}):
                x = (((board.dealer - 1) + i) % 4 - 1.5) * 60 - 10
                text = font.render(b.bid[0], 1, (0, 0, 0))
                image = eval(b.bid[1])
                original_image_rect = image.get_rect()
                # Scale the image to 80% of its original size
                scale_factor = 0.8
                new_image_size = (int(original_image_rect.width * scale_factor), int(original_image_rect.height * scale_factor))
                image = pygame.transform.scale(image, new_image_size)
                win.blit(image, (round(win.get_width() / 2 + x - image.get_width() / 2) + text.get_width() + 4, round(y - image.get_height() / 2)))
                win.blit(text, (round(win.get_width() / 2 + x - text.get_width() / 2), round(y - text.get_height() / 2)))
            else:
                x = (((board.dealer - 1) + i) % 4 - 1.5) * 60
                text = font.render(b.bid.replace("PASS","P"), 1, (0, 0, 0))                 
                win.blit(text, (round(win.get_width() / 2 + x - text.get_width() / 2), round(y - text.get_height() / 2)))
    # Buttons
    for btn in buttons:
        btn.draw(win)
    # Updating pygame window
    pygame.display.update()


def redraw_playing(win, font, font2, buttons, table, board, user):
    """
    Drawing all elements during playing phase.
    :param win: pygame Surface instance
    :param font: pygame Surface instance
    :param font2: pygame Surface instance
    :param buttons: list
    :param table: Table instance
    :param board: Board instance
    :param user: Player instance
    :return: None
    """
    # Drawing seats
    redraw_sitting(win, font, table, user)
    # Drawing table status
    # game_ready_text = font2.render("Game", 1, (0, 0, 0))
    #win.blit(game_ready_text, (round(win.get_width() / 2 - game_ready_text.get_width() / 2),
    #                           round(win.get_height() / 2 - game_ready_text.get_height() / 2)))
    # Drawing cards
    draw_cards(win, font, table, board, user)
    # Drawing info about declarer, final contract and taken tricks by each side
    if board.declarer[1] == [0, 2]:
        contract_text = font.render(f"Kontrakt {board.winning_bid} - NS", 1, (0, 0, 0))
        if board.declarer[0] == 0:
            declarer_text = font.render("Declarer: S", 1, (0, 0, 0))
        else:
            declarer_text = font.render("Declarer: N", 1, (0, 0, 0))
    else:
        contract_text = font.render(f"Kontrakt {board.winning_bid} - EW", 1, (0, 0, 0))
        if board.declarer[0] == 1:
            declarer_text = font.render("Declarer: W", 1, (0, 0, 0))
        else:
            declarer_text = font.render("Declarer: E", 1, (0, 0, 0))
    win.blit(contract_text, (920, 880))
    win.blit(declarer_text, (920, 920))
    tricks_text = font.render(f"NS: {board.tricks[0]}  EW: {board.tricks[1]}", 1, (0, 0, 0))
    win.blit(tricks_text, (920, 840))

    # Drawing the trick
    for i, p in enumerate(table.players):
        card = i
        i -= user.position
        if i < 0:
            i += 4
        if board.trick[card]:
            # Rotating the trick to client
            if i == 0:
                board.trick[card].draw(win, round(win.get_width() / 2 - 50), 600, True)
            elif i == 2:
                board.trick[card].draw(win, round(win.get_width() / 2 - 50), 330, True)
            elif i == 1:
                board.trick[card].draw(win, 400, round(win.get_height() / 2 - 153 / 2), True)
            else:
                board.trick[card].draw(win, 700, round(win.get_height() / 2 - 153 / 2), True)
    # Buttons
    for btn in buttons:
        btn.draw(win)
    # Updating pygame window
    pygame.display.update()


def redraw_score(win, font, font2, buttons, table, board, user):
    """
    Drawing score after finished board.
    :param win: pygame Surface instance
    :param font: pygame Surface instance
    :param font2: pygame Surface instance
    :param buttons: list
    :param table: Table instance
    :param board: Board instance
    :param user: Player instance
    :return: None
    """
    # Drawing seats
    redraw_sitting(win, font, table, user)
    # Drawing title with table ID
    table_text = font2.render(f"Board {table.id}", 1, (0, 0, 0))
    win.blit(table_text, (round(win.get_width() / 2 - table_text.get_width() / 2), 10))
    # Drawing score
    if not board.declarer:
        score_text = font2.render("Pass out", 1, (0, 0, 0))
    else:
        if board.declarer[1] == [0, 2]:
            score_text = font2.render(f"NS {board.result}, {board.score}", 1, (0, 0, 0))
        else:
            score_text = font2.render(f"EW {board.result}, {board.score}", 1, (0, 0, 0))
    win.blit(score_text, (round(win.get_width() / 2 - score_text.get_width() / 2),
                          round(win.get_height() / 2 - score_text.get_height() / 2)))
    # Updating pygame window
    pygame.display.update()
