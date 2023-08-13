import datetime
import sys

from collections import deque
from typing import NamedTuple


class Deal(NamedTuple):
    dealer: str
    vulnerable: str
    hands: str

def pbn_generator(boards):

    def print_deal(board_number, dlr, vul, deal):
        print('[Event "?"]')
        print('[Site "?"]')
        print(f'[Date "{datetime.datetime.now().date().isoformat()}"]')
        print(f'[Board "{board_number}"]')
        print('[West "?"]')
        print('[North "?"]')
        print('[East "?"]')
        print('[South "?"]')
        print('[Scoring "IMP"]')
        print(f'[Dealer "{dlr}"]')
        print(f'[Vulnerable "{vul}"]')
        print(f'[Deal "{deal}"]')
        print('[Declarer "?"]')
        print('[Result "?"]\n')

    for i in range(len(boards)):
        parts = boards[i].split()
        dealer_str = parts[0]
        vulnerable_str = parts[1].replace("-","")
        deal_str = " ".join(parts[2:])
        print_deal(i + 1, dealer_str, vulnerable_str, f'N:{deal_str}')
        

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python ben2pbn.py input.ben")
        sys.exit(1)
    
    input_file = sys.argv[1]
    try:
        with open(input_file, "r", encoding='utf-8') as file:  # Open the input file with UTF-8 encoding
            lines = file.readlines()
            lines = [line.strip() for line in lines]
        pbn_generator(lines)
    except Exception as ex:
        print('Error:', ex)
        raise ex
