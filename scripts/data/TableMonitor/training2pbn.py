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
        print('[Event "BEN Match"]')
        print(f'[Date "{datetime.datetime.now().date().isoformat()}"]')
        print(f'[Board "{board_number}"]')
        print('[Scoring "IMP"]')
        print(f'[Dealer "{dlr}"]')
        print(f'[Vulnerable "{vul}"]')
        print(f'[Deal "{deal}"]')
        print("")

    dealnumber = 0
    for i in range(len(boards) // 2):
        deal_str = boards[i*2]
        parts = boards[i*2+1].split()
        dealer_str = parts[0]
        vulnerable_str = parts[1].replace("-","")
        if (i % 2 == 0):
            dealnumber = dealnumber + 1
            print_deal(dealnumber, dealer_str, vulnerable_str, f'N:{deal_str}')
        

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python training2pbn.py training.txt")
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
