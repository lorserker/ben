import sys
from gevent import monkey
monkey.patch_all()

# Patching to suppress the warning about invalid HTTP versions
monkey.patch_all(warn_on_full_argv=False, warn_on_stopping=False)

from bottle import Bottle, run, static_file, redirect, template, request, response, HTTPError
import bottle
bottle.BaseRequest.MEMFILE_MAX = 5 * 1024 * 1024 

import shelve
import uuid
import json
import os
import argparse
import datetime
import numpy as np
from urllib.parse import parse_qs
from urllib.parse import quote
import re


app = Bottle()
os.getcwd()

BUNDLE_TEMP_DIR = ''

try:
    if getattr(sys, 'frozen') and hasattr(sys, '_MEIPASS'):
        BUNDLE_TEMP_DIR = sys._MEIPASS
        #Modify template
        bottle.TEMPLATE_PATH.insert(0,BUNDLE_TEMP_DIR + '/views')
except:
    BUNDLE_TEMP_DIR= '' 

def extract_value(s: str) -> str:
    return s[s.index('"') + 1 : s.rindex('"')]

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

def is_valid_deal_id(deal_id):
    # Check if the deal_id is a valid hexadecimal string
    return bool(re.match('^[0-9a-fA-F]{32}$', deal_id))

def hand_as_string(hand):
    s = ""
    for i in range(4):
        for j in range(4):
            s += hand[i].suit[3 - j]
            if j == 3:  # Checks if we've reached the last suit
                s += " "
            else:
                s += "."
    return s

def encode_board(hand, dealer, vulnerability, deal_number):
    board_extension = ((deal_number - 1) // 16) % 16
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

def transform_hand(hands):
    hand = [TypeHand() for _ in range(4)]

    for i in range(4):
        suits = hands[i].split('.')
        hand[i].suit[C_CLUBS] = suits[3]
        hand[i].suit[C_DIAMONDS] = suits[2]
        hand[i].suit[C_HEARTS] = suits[1]
        hand[i].suit[C_SPADES] = suits[0]
    
    return hand


def decode_board(encoded_str_deal):
    # Initialize the hand
    hand = [TypeHand() for _ in range(4)]

    board_extension = int(encoded_str_deal[0], 16)
    number = int(encoded_str_deal[1], 16)
    dealer = number // 4
    vulnerable = number % 4
    deal_no = board_extension * 16 + board[dealer][vulnerable]
    encryption_byte = board[dealer][vulnerable]

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
    dealer = "NESW"[dealer]
    vulnerable = ['None', 'N-S', 'E-W', 'Both'][vulnerable]
    return hand, dealer, vulnerable, deal_no

def parse_lin(lin):
    rx_hand = r'S(?P<S>[2-9A,K,Q,J,T]*?)H(?P<H>[2-9A,K,Q,J,T]*?)D(?P<D>[2-9A,K,Q,J,T]*?)C(?P<C>[2-9A,K,Q,J,T]*?)$'

    lin_vuln = re.findall(r'sv\|(.)\|', lin)[0]
    vuln = "None"
    if lin_vuln == 'n':
        vuln = "N-S"
    elif lin_vuln == 'e':
        vuln = "E-W"
    elif lin_vuln == 'b':
        vuln = "Both"

    lin_deal = re.findall(r'(?<=md\|)(.*?)(?=\|)', lin)[0]
    dealer = {'1': 'S', '2': 'W', '3': 'N', '4': 'E'}[lin_deal[0]]
    lin_hands = lin_deal[1:].split(',')

    hd_south = re.search(rx_hand, lin_hands[0]).groupdict()
    hd_west = re.search(rx_hand, lin_hands[1]).groupdict()
    hd_north = re.search(rx_hand, lin_hands[2]).groupdict()

    if lin_hands[3]:
        hd_east = re.search(rx_hand, lin_hands[3]).groupdict()
    else:
        def seen_cards(suit):
            return set(hd_south[suit]) | set(hd_west[suit]) | set(hd_north[suit])

        hd_east = {suit: set('AKQJT98765432') - seen_cards(suit) for suit in 'SHDC'}

    def to_pbn(hd):
        return '.'.join([''.join(list(hd[suit])) for suit in 'SHDC'])

    hands = [to_pbn(hd) for hd in [hd_north, hd_east, hd_south, hd_west]]
    # Pattern to find "Board <number>" and extract the number
    pattern = r'Board (\d+)'

    # Using re.search to find the pattern in the text
    match = re.search(pattern, lin)

    if match:
        board_no = match.group(1)  # Extracting the matched board number
    else:
        board_no = ""

    return dealer, vuln, hands, board_no

def parse_bsol(url):
    query_params = parse_qs(url.split('?')[1])
    # Extract the required values
    board = query_params.get('board', [])[0]
    dealer = query_params.get('dealer', [])[0]
    vuln_str = query_params.get('vul', [])[0]
    vulnerable = {'NS': 'N-S', 'EW': 'E-W', 'All': 'Both'}.get(vuln_str, vuln_str)

    hands = []
    # Concatenate the values from North, South, East, and West
    hands.append(query_params.get('North', [])[0])
    hands.append(query_params.get('South', [])[0])
    hands.append(query_params.get('East', [])[0])
    hands.append(query_params.get('West', [])[0])

    hands = " ".join(hands)
    print(hands)

    return dealer, vulnerable, hands, board
    
def parse_pbn(fin):
    for line in fin:
        if line.startswith('[Dealer'):
            dealer = extract_value(line)
        if line.startswith('[Vulnerable'):
            vuln_str = extract_value(line)
            vulnerable = {'NS': 'N-S', 'EW': 'E-W', 'All': 'Both'}.get(vuln_str, vuln_str)
        if line.startswith('[Board'):
            board = extract_value(line)
            if not board.isdigit():
                last_space_index = board.rfind(' ')
                board = board[last_space_index + 1:]
        if line.startswith('[Deal '):
            hands_pbn = extract_value(line)
            [seat, hands] = hands_pbn.split(':')
            hands_nesw = [''] * 4
            first_i = 'NESW'.index(seat)
            for hand_i, hand in enumerate(hands.split()):
                hands_nesw[(first_i + hand_i) % 4] = hand
        else:
            continue

    return dealer, vulnerable, hands, board

parser = argparse.ArgumentParser(description="Appserver")
parser.add_argument("--host", default="localhost", help="Hostname for appserver")
parser.add_argument("--port", type=int, default=8080, help="Port for appserver")
parser.add_argument("--db", default="../gamedb", help="Db for appserver")

args = parser.parse_args()

port = args.port
DB_NAME = os.getcwd() + "/" + args.db
print("Reading deals from: "+DB_NAME)

script_dir = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index(): 
    return template('index.tpl') 

@app.route('/error')
def error_page():
    error_message = request.query.message
    # Render your error page template here, passing error_message to the template
    return f"<h3>{error_message}</h3>"

@app.route('/submit', method="POST")
def index(): 
    url = None
    north = request.forms.get('N')
    east = request.forms.get('E')
    south = request.forms.get('S')
    west = request.forms.get('W')    
    human = request.forms.get('H')    
    autocomplete = request.forms.get('A')
    name = request.forms.get('name')
    timeout = request.forms.get('T')
    cont = request.forms.get('C')
    rotate = request.forms.get('R')
    visible = request.forms.get('V')
    player = ""
    if north: player += "&N=x"
    if east: player += "&E=x"
    if south: player += "&S=x"
    if west: player += "&W=x"
    if human: player += "&H=x"
    if autocomplete: player += "&A=x"
    if name: player += f"&name={name}"
    if timeout: player += f"&T={timeout}"
    if cont: player += "&C=x"
    if rotate: player += "&R=x"
    if visible: player += "&V=x"
    dealtext = request.forms.get('dealtext')
    if dealtext:
        dealer = request.forms.get('dealer')
        board_no = request.forms.get('board')
        vulnerable = request.forms.get('vulnerable')
        dealtext = dealtext.upper()
        url = f'/app/bridge.html?deal=(%27{dealtext}%27, %27{dealer} {vulnerable}%27){player}&board_no={board_no}'
    
    dealpbn = request.forms.get('dealpbn')
    if dealpbn:
        try:
            dealpbn = request.forms.get('dealpbn')
            dealer, vulnerable, hands, board_no = parse_pbn(dealpbn.splitlines())
            url = f'/app/bridge.html?deal=(%27{hands}%27, %27{dealer} {vulnerable}%27){player}&board_no={board_no}'
        except Exception as e:
            error_message = f'Error parsing PBN-input. {e}'
            print(error_message)
            print(dealpbn)
            encoded_error_message = quote(error_message)
            redirect(f'/error?message={encoded_error_message}')

    dealbsol = request.forms.get('dealbsol')
    if dealbsol:
        dealbsol = request.forms.get('dealbsol')
        dealer, vulnerable, hands, board_no = parse_bsol(dealbsol)
        url = f'/app/bridge.html?deal=(%27{hands}%27, %27{dealer} {vulnerable}%27){player}&board_no={board_no}'

    deallin = request.forms.get('deallin')
    if deallin:
        query_params = deallin.split('?')
        deallinparsed = parse_qs(query_params[-1])
        try:
            lin = deallinparsed["lin"]
            dealer, vulnerable, hands, board_no = parse_lin(lin[0])
        except KeyError:
            try:
                dealer, vulnerable, hands, board_no = parse_lin(deallin)
            except Exception as e:
                error_message = f'Error parsing LIN-input. {e}'
                print(error_message)
                print(deallin)
                encoded_error_message = quote(error_message)
                redirect(f'/error?message={encoded_error_message}')

        hands = " ".join(hands)
        url = f'/app/bridge.html?deal=(%27{hands}%27, %27{dealer} {vulnerable}%27){player}&board_no={board_no}'

    dealbba = request.forms.get('dealbba')
    if dealbba:
        hand, dealer, vulnerable, board_no = decode_board(dealbba)
        deal_as_str = hand_as_string(hand)
        url = f'/app/bridge.html?deal=(%27{deal_as_str}%27, %27{dealer} {vulnerable}%27){player}&board_no={board_no}'
    if url:
        redirect(url)
    else:
        board_no = request.forms.get('board')
        redirect(f"/app/bridge.html?board_no={board_no}{player}")
    

@app.route('/home')
def home():
    deals = []
    with shelve.open(DB_NAME) as db:
        deal_items = sorted(list(db.items()), key=lambda x: x[1]['timestamp'], reverse=True)
        for deal_id, deal in deal_items:
            board_no_ref = ""
            board_no_index = ""
            if 'feedback' in deal: feedback = deal['feedback']
            else: feedback = ""
            if 'quality' in deal: quality = deal['quality']
            else: quality = 'ok'
            if 'board_number' in deal and deal['board_number'] is not None:
                board_no_ref = f"&board_number={deal['board_number']}"
                board_no_index = f"Board:{deal['board_number']}"
            vulnerable = C_NONE
            if (deal["vuln_ns"]) and (deal["vuln_ew"]): vulnerable = C_BOTH
            else:
                if deal["vuln_ns"]: vulnerable = C_NS
                if deal["vuln_ew"]: vulnerable = C_WE
            encoded_str_deal = encode_board(transform_hand(deal["hands"].split(" ")), deal["dealer"], vulnerable, int(deal['board_number']) if deal['board_number'] is not None else 0)                
            # Trick winners are relative to declarer so 1 and 3 are declarer and dummy
            tricks = len(list(filter(lambda x: x % 2 == 1, deal['trick_winners'])))

            if 'claimed' in deal:
                if 'claimedbydeclarer' in deal and deal['claimedbydeclarer']:
                    tricks += deal['claimed']
                else:
                    tricks += 13 - len(deal['trick_winners'])-deal['claimed']

            if deal['contract'] is not None:
                deals.append({
                    'board_no_index': board_no_index,
                    'deal_id': deal_id,
                    'board_no_ref': board_no_ref,
                    'contract': deal['contract'],
                    'trick_winners_count': tricks,
                    'delete_url': f"/api/delete/deal/{deal_id}",
                    'bba': encoded_str_deal,
                    'feedback':feedback,
                    'quality': quality
                })
            else:
                deals.append({
                    'board_no_index': board_no_index,
                    'deal_id': deal_id,
                    'board_no_ref': board_no_ref,
                    'contract': "All Pass",
                    'delete_url': f"/api/delete/deal/{deal_id}",
                    'bba': encoded_str_deal,
                    'feedback':feedback,
                    'quality': quality
                })

    return template('home.tpl', deals=deals)

@app.route('/app/<filename>')
def frontend(filename):
    if '?' in filename:
        filename = filename[:filename.index('?')]

    file_path = os.path.join(script_dir, '')    
    return static_file(filename, root=file_path)

@app.route('/favicon.ico')
def frontend():
    filename = 'favicon.ico'
    file_path = os.path.join(script_dir, '')    
    return static_file(filename, root=file_path)

@app.route('/api/deals/<deal_id>')
def deal_data(deal_id):
    print("Getting:", deal_id)
    try:
        db = shelve.open(DB_NAME)
        deal = db[deal_id]
        db.close()

        return json.dumps(deal)
    except KeyError:
        print("Deal not found")
        raise HTTPError(404, "Deal not found")

@app.route('/api/delete/deal/<deal_id>')
def delete_deal(deal_id):
    print("Deleting:", deal_id)
    if not is_valid_deal_id(deal_id):
        print("Invalid deal ID")
        raise HTTPError(400, "Invalid deal ID")
    try:
        db = shelve.open(DB_NAME)
        db.pop(deal_id)
        db.close()
        print("Returning to home")

        redirect('/home')
    except KeyError:
        print("Deal not found")
        raise HTTPError(404, "Deal not found")

@app.route('/api/save/deal', method='POST')
def save_deal():
    data_dict = request.json  # Get JSON data from the request body
    if data_dict:
        db = shelve.open(DB_NAME)
        db[uuid.uuid4().hex] = data_dict
        db.close()
        response.status = 200  # HTTP status code: 200 OK
        response.headers['Content-Type'] = 'application/json'  # Set response content type
        return json.dumps({'message': 'Deal saved successfully'})
    else:
        print("Invalid data received")
        raise HTTPError(400, "Invalid data received")
    
host = args.host
print(f'http://{host}:{port}/home')

run(app, host=host, port=port, server='gevent', log=None)
