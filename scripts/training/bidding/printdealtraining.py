import itertools
import numpy as np

class TypeHand:
    def __init__(self):
        self.suit = [""] * 4

board = np.zeros((4, 4), dtype=int)
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

def encode_board(hand, dealer, vulnerability, deal_number):
    board_extension = ((deal_number - 1) // 16) % 16
    deal_str = format(board_extension, 'x') + format(dealer * 4 + vulnerability, 'x')
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
        deal_str += str_cards.upper()

    return deal_str.upper()  # Convert the entire string to uppercase for consistency with VBA output

def transform_hand(hands):
    hand = [TypeHand() for _ in range(4)]

    for i in range(4):
        suits = hands[i].split('.')
        hand[i].suit[C_CLUBS] = suits[3]
        hand[i].suit[C_DIAMONDS] = suits[2]
        hand[i].suit[C_HEARTS] = suits[1]
        hand[i].suit[C_SPADES] = suits[0]
    
    return hand

def generate_html_card(suit, cards):
    html = f"<div class='suit'><span>{suit}</span>"
    for card in cards:
        html += f"{card}"
    html += "</div>"
    return html

def generate_html_deal(line, board_number, bidding, bidding2):
    print(f"Generating board {board_number}")
    parts = line.split()
    dealer = parts[0]
    vuln= parts[1]
    cards = parts[2:]
    vulnerable = C_NONE
    if vuln == "All" : vulnerable = C_BOTH
    if vuln == "All" : vulnerable = C_NS
    if vuln == "All" : vulnerable = C_WE
    encoded_str_deal = encode_board(transform_hand(cards), "NESW".index(dealer), vulnerable, board_number)                

    html = f"""
    <!DOCTYPE html>
    <html lang='en'>
    <head>
        <meta charset='utf-8'>
        <title>Match deal</title>
        <link rel='stylesheet' href='viz.css'>
        <script src="viz.js"></script>  
    </head>
    <body>
        <div class='center'>
        <div id='deal'>
            <div id='dealer-vuln'>
                <div id='vul-north' class='{"red" if vulnerable in ('N-S', 'Both') else 'white'}'>
                    {"<span class='dealer'>N</span>" if dealer == 'N' else ''}
                </div>
                <div id='vul-east' class='{"red" if vulnerable in ('E-W', 'Both') else 'white'}'>
                    {"<span class='dealer'>E</span>" if dealer == 'E' else ''}
                </div>
                <div id='vul-south' class='{"red" if vulnerable in ('N-S', 'Both') else 'white'}'>
                    {"<span class='dealer'>S</span>" if dealer == 'S' else ''}
                </div>
                <div id='vul-west' class='{"red" if vulnerable in ('E-W', 'Both') else 'white'}'>
                    {"<span class='dealer'>W</span>" if dealer == 'W' else ''}
                </div>
                <div id='boardno'>
                    {board_number}
                </div>
            </div>
            <div id='north'>
                {generate_html_card('&spades;', cards[0].split('.')[0])}
                {generate_html_card('<span class="font-red">&hearts;</span>', cards[0].split('.')[1])}
                {generate_html_card('<span class="font-red">&diams;</span>', cards[0].split('.')[2])}
                {generate_html_card('&clubs;', cards[0].split('.')[3])}
            </div>
            <div id='west'>
                {generate_html_card('&spades;', cards[3].split('.')[0])}
                {generate_html_card('<span class="font-red">&hearts;</span>', cards[3].split('.')[1])}
                {generate_html_card('<span class="font-red">&diams;</span>', cards[3].split('.')[2])}
                {generate_html_card('&clubs;', cards[3].split('.')[3])}
            </div>
            <div id='east'>
                {generate_html_card('&spades;', cards[1].split('.')[0])}
                {generate_html_card('<span class="font-red">&hearts;</span>', cards[1].split('.')[1])}
                {generate_html_card('<span class="font-red">&diams;</span>', cards[1].split('.')[2])}
                {generate_html_card('&clubs;', cards[1].split('.')[3])}
            </div>
            <div id='south'>
                {generate_html_card('&spades;', cards[2].split('.')[0])}
                {generate_html_card('<span class="font-red">&hearts;</span>', cards[2].split('.')[1])}
                {generate_html_card('<span class="font-red">&diams;</span>', cards[2].split('.')[2])}
                {generate_html_card('&clubs;', cards[2].split('.')[3])}
            </div>
        </div>
        <!--
        <a href="http://127.0.0.1:8080/app/bridge.html?deal=('{' '.join(cards)}', '{dealer} {vulnerable}')&P=0&A=x&board_no={board_number}"> Se it played (no search for NS) </a><br>
        <a href="http://127.0.0.1:8080/app/bridge.html?deal=('{' '.join(cards)}', '{dealer} {vulnerable}')&P=1&A=x&board_no={board_number}"> Se it played (no search for EW) </a><br>
        <a href="http://127.0.0.1:8080/app/bridge.html?deal=('{' '.join(cards)}', '{dealer} {vulnerable}')&P=4&A=x&board_no={board_number}"> Se it played (Search for both) </a><br>
        -->
        """
    if board_number == 0:
        html += f"""<a href="index.html">Home</a>&nbsp;"""
    else:
        html += f"""<a href="Board{board_number-1}.html">Previous</a>&nbsp;"""
    if board_number == 1000:
        html += f"""<a href="index.html">Home</a>"""
    else:
        html += f"""<a href="Board{board_number+1}.html">Next</a><br><br>"""
    html += f"""    Code for BBA: {encoded_str_deal}
        
        <br>
        <div id="auction"></div>
        <div id="auction2"></div>
        </div>
        <script>
            let auction = new Auction({'NESW'.index(dealer)}, {bidding[:-1]}, {bidding.pop()})
            auction.render(document.getElementById("auction"))
            auction = new Auction({'NESW'.index(dealer)}, {bidding2[:-1]}, {bidding2.pop()})
            auction.render(document.getElementById("auction2"))
        </script>
    </body>
    </html>"""

    filename = f"./html/Board{board_number}.html"
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(html)


# Read the file and generate HTML for each line
with open('sampling_1000.prt', 'r') as file:
    lines = list(file)
    for board_number, (line, line1, line2, line3) in enumerate(zip(lines[::4], lines[1::4], lines[2::4], lines[3::4]), start=1):
        bidding = [{'bid': bid} for bid in line2.split()]
        bidding2 = [{'bid': bid} for bid in line3.split()]
        generate_html_deal(line1, board_number, bidding, bidding2)
