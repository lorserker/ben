import sys
from collections import deque
from typing import NamedTuple

class Deal(NamedTuple):
    dealer: str
    vulnerable: str
    hands: str


def load(fin):
    boards = [] 
    dealer, vulnerable = None, None
    for line in fin:
        if line.startswith('[Dealer'):
            dealer = extract_value(line)
        elif line.startswith('[Vulnerable'):
            vuln_str = extract_value(line)
            vulnerable = {'NS': 'N-S', 'EW': 'E-W', 'All': 'Both'}.get(vuln_str, vuln_str)
        elif line.startswith('[Deal'):
            hands_pbn = extract_value(line)
            [seat, hands] = hands_pbn.split(':')
            hands_nesw = [''] * 4
            first_i = 'NESW'.index(seat)
            for hand_i, hand in enumerate(hands.split()):
                hands_nesw[(first_i + hand_i) % 4] = hand
            boards.append("'" + ' '.join(hands_nesw) + "','" + dealer + " " + vulnerable + "'") 
        else:
            continue
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
            print(boards)
            for board in boards:
                print(board)
                file.write(board + "\n")

    except Exception as ex:
        print('Error:', ex)
        raise ex
