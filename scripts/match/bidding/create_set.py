import sys
sys.path.append('../../../src')

from collections import deque

import deck52


def generate(n_boards):
    dealer = list('NESW')
    vuln = deque(['None', 'N-S', 'E-W', 'Both'])
    
    for i in range(n_boards):
        deal_str = deck52.random_deal()
        
        if i % 4 == 0 and i > 0:
            vuln.append(vuln.popleft())
    
        yield dealer[i % 4], vuln[i % 4], deal_str


if __name__ == '__main__':
    n_boards = int(sys.argv[1])

    for dealer, vuln, hands_str in generate(n_boards):
        print(f'{dealer} {vuln} {hands_str}')
        