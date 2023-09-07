import sys

def convert_deal_str(deal_str):
    '''
    converts between the following two:
    W:KQJ952.98.AQ9.95 6.AQ6.T8765.AKJ3 T874.J73.K3.8742 A3.KT542.J42.QT6
     S6 HAQ6 DT8765 CAKJ3  ST874 HJ73 DK3 C8742 SA3 HKT542 DJ42 CQT6  SKQJ952 H98 DAQ9 C95
    '''
    assert deal_str.startswith('W:')
    hands = deal_str[2:].strip().split()
    hands = [hands[1], hands[2], hands[3], hands[0]]
    return '  '.join(map(convert_hand_str, hands))

def convert_hand_str(hand_str):
    '''
    converts from:
    KQJ952.98.AQ9.95
    to
    SKQJ952 98 AQ9 95
    '''
    suits = hand_str.split('.')
    return ' '.join(['S' + suits[0], 'H' + suits[1], 'D' + suits[2], 'C' + suits[3]])

def convert_auction_str(auction_str):
    return auction_str.strip().replace('PP', 'P').replace('DD', 'X').replace('RR', 'XX')

jack_vuln_lookup = {
    '-': 'None',
    'NS': 'N-S',
    'EW': 'E-W',
    'ALL': 'Both'
}

def parse_dealer_vuln(s):
    parts = s.split()
    dealer = parts[0]
    vuln = jack_vuln_lookup[parts[1]]
    return (dealer, vuln)

if __name__ == '__main__':
    dealer, vuln = None, None
    for i, line in enumerate(sys.stdin):
        if i % 4 == 0:
            print(' ' + convert_deal_str(line))
        if i % 4 == 1:
            dealer, vuln = parse_dealer_vuln(line)
        if i % 4 == 2:
            auction_str = convert_auction_str(line)
            print(' %s %s %s' % (dealer, vuln, auction_str))
