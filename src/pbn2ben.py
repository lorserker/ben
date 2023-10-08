import sys
import re
from collections import deque
from typing import NamedTuple

class Deal(NamedTuple):
    dealer: str
    vulnerable: str
    hands: str


def load(fin):
    boards = [] 
    auction_lines = []
    inside_auction_section = False
    dealer, vulnerable = None, None
    for line in fin:
        if line.startswith("% PBN") or line == "\n":
            if dealer != None:
                board = {
                    'deal': ' '.join(hands_nesw),      
                    'auction': dealer + " " + vulnerable + " " + ' '.join(auction_lines)
                }
                boards.append(board)            
                auction_lines = []
                dealer= None
        if line.startswith('[Dealer'):
            dealer = extract_value(line)
        if line.startswith('[Vulnerable'):
            vuln_str = extract_value(line)
            vulnerable = {'NS': 'N-S', 'EW': 'E-W', 'All': 'Both'}.get(vuln_str, vuln_str)
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
    if dealer != None:
        board = {
            'deal': ' '.join(hands_nesw),      
            'auction': dealer + " " + vulnerable + " " + ' '.join(auction_lines)
        }
        boards.append(board)      
    return boards

def extract_value(s: str) -> str:
    return s[s.index('"') + 1 : s.rindex('"')]

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python pbn2ben.py input_file.pbn")
        sys.exit(1)
    
    input_file = sys.argv[1]
    try:
        with open(input_file, "r", encoding='utf-8') as file:  # Open the input file with UTF-8 encoding
            lines = file.readlines()
        
        with open('input.ben', 'w', encoding='utf-8') as file:  # Open the output file with UTF-8 encoding
            boards = load(lines)
            for board in boards:
                file.write(board['deal'] + "\n" + board['auction'] + "\n")
            print("File input.ben generated")
    except Exception as ex:
        print('Error:', ex)
        raise ex
