import sys

def gib_iter(fin):
    deal = {}
    for i, line in enumerate(fin):
        line = line.strip()
        if i % 9 == 0:
            deal['W'] = line[2:]
        elif i % 9 == 1:
            deal['N'] = line[2:]
        elif i % 9 == 2:
            deal['E'] = line[2:]
        elif i % 9 == 3:
            deal['S'] = line[2:]
        elif i % 9 == 4:
            deal['dealer_vuln'] = line
        elif i % 9 == 5:
            deal['outcome'] = line
        elif i % 9 == 6:
            deal['auction'] = line
        elif i % 9 == 7:
            deal['play'] = line
        elif i % 9 == 8:
            yield deal
            deal = {}

def to_training(gib_iter):
    for i, deal in enumerate(gib_iter):
        if i % 10000 == 0:
            sys.stderr.write('%d\n' % i)
        auction =  deal['dealer_vuln'].replace("-","None").replace("NS","N-S").replace("EW","E-W").replace("ALL","Both") + " " + deal['auction'].replace("."," ")

        print('%s %s %s %s' % (deal['N'], deal['E'], deal['S'], deal['W']))
        print(auction)


if __name__ == '__main__':
    to_training(gib_iter(sys.stdin))