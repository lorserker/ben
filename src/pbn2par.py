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
    dealer, vulnerable, par_contract = None, None, None
    for line in fin:
        if line.startswith("% PBN") or line == "\n":
            if dealer != None:
                board = {
                    'deal': ' '.join(hands_nesw),      
                    'auction': dealer + " " + vulnerable + " " + ' '.join(auction_lines),
                    'par' : par_contract + ' ' + side + ' ' + str(tricks)
                }
                boards.append(board)            
                auction_lines = []
                dealer= None
                vulnerable = None
                par_contract = None
        if line.startswith('[Dealer'):
            dealer = extract_value(line)
        if line.startswith('[OptimumResult '):
            optimum_result = extract_value(line)
            parts = re.match(r'(\S{2,3})\s*([NEWS]{1,2})(\D?\d)?', optimum_result).groups()
            par_contract = parts[0].replace('NT', 'N')
            side = parts[1]
            if parts[2] is None:
                tricks = int(par_contract[0]) + 6
            else:
                tricks = int(parts[2]) + int(par_contract[0]) + 6
        if line.startswith('[OptimumScore '):
            optimum_result = extract_value(line)
            parts = re.match(r'(\S{2,3})\s*([NEWS]{1,2})(\D?\d)?', optimum_result).groups()
            par_contract = parts[0].replace('NT', 'N')
            side = parts[1]
            if parts[2] is None:
                tricks = int(par_contract[0]) + 6
            else:
                tricks = int(parts[2]) + int(par_contract[0]) + 6
        if line.startswith('[ParScore'):
            optimum_result = extract_value(line)
            parts = optimum_result.split('.')
            par_contract = parts[0].replace('NT', 'N')
            side = parts[2]
            tricks = parts[1]
        if line.startswith('[ParContract'):
            optimum_result = extract_value(line).replace("=",'')
            parts = re.match(r'([NEWS]{1,2})\s*(\S{2,3})(\D?\d)?', optimum_result).groups()
            par_contract = parts[1].replace('NT', 'N').replace('+','')
            side = parts[0]
            if parts[2] is None:
                tricks = int(par_contract[0]) + 6
            else:
                tricks = int(parts[2]) + int(par_contract[0]) + 6
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
            'auction': dealer + " " + vulnerable + " " + ' '.join(auction_lines),
            'par' : par_contract + ' ' + side + ' ' + str(tricks)
        }
        boards.append(board)      
    return boards

def extract_value(s: str) -> str:
    return s[s.index('"') + 1 : s.rindex('"')]

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python pbn2par.py input_file.pbn")
        sys.exit(1)
    
    input_file = sys.argv[1]
    try:
        with open(input_file, "r", encoding='utf-8') as file:  # Open the input file with UTF-8 encoding
            lines = file.readlines()
        
        with open('input.par', 'w', encoding='utf-8') as file:  # Open the output file with UTF-8 encoding
            boards = load(lines)
            for board in boards:
                file.write(board['deal'] + "\n" + board['auction'] + "\n" + board['par'] + "\n")
            print("File input.par generated")
    except Exception as ex:
        print('Error:', ex)
        raise ex
