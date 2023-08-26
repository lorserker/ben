import sys

def main():
    auction = []

    for line in sys.stdin:
        line = line.strip()
        if any((line.startswith('meaning'), line.startswith('min'), line.startswith('max'))):
            continue

        if line.startswith('BID ='):
            bid_i = int(line[5:])
            if bid_i == 0:
                auction.append('P')
            elif bid_i == 1:
                auction.append('X')
            elif bid_i == 2:
                auction.append('XX')
            elif bid_i >= 5:
                bid = f'{bid_i // 5}{"CDHSN"[bid_i % 5]}'
                auction.append(bid)
            else:
                raise AttributeError(f'Unexpected bid {bid_i}')
            continue

        # we are getting a new deal
        parts = line.split()
        dealer_vuln = ' '.join(parts[:2])
        hands = ' '.join(parts[2:])
        
        if len(auction) > 0:
            sys.stdout.write(' '.join(auction))
            sys.stdout.write('\n')
            auction = []
        
        print(hands)
        sys.stdout.write(f'{dealer_vuln} ')
    
    sys.stdout.write(' '.join(auction))
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
