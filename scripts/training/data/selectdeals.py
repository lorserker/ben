def is_valid_bid(line):
    line = line.replace(" ", "")
    try:
        index = line.index("2CP2HP2SP4H")
        try:
            index = line.index("1")
            return False
        except:
            return True
    except:
        return False

def get_hcp(hand):
    hcp = {'A': 4, 'K': 3, 'Q': 2, 'J': 1}
    return sum([hcp.get(c, 0) for c in hand])

def opening_with_pass(line):
    bids = line2.split()

    return bids[2] == "P"
def opening_with_2D(line):
    bids = line2.split()

    return bids[2] == "2D"
# Define a dictionary to map directions to index updates
direction_to_index = {'N': 0, 'E': 1, 'S': 2, 'W': 3}

# Read the file and validate bid for each board
with open('GIB-Thorvald-2024-08-09.ben', 'r') as file:
    lines = list(file)
    opening = 0
    passing = 0
    for board_number, (line1, line2) in enumerate(zip(lines[::2], lines[1::2]), start=1):
        hands = line1.split()
        bids = line2.split()
        dealer = direction_to_index[bids[0]]
        hcp = get_hcp(hands[dealer])
        # We only want hcp in range 6-11
        if hcp < 6:
            continue
        if hcp > 11:
            continue
        suits = hands[dealer].split('.')
        # At least 6 diamonds
        if len(suits[2]) < 6:
            continue
        print(line1, end="")
        print(line2, end="")
        continue
        if (opening_with_pass(line2)):
            #for suit in suits:
            if len(suits[2]) > 5:
                passing += 1
                print("P",get_hcp(hands[dealer]),hands[dealer])
                #print("P ",hands[dealer])
                #print(line1, end="")
                #print(line2, end="")
        if (opening_with_2D(line2)):
            print("2D",hcp,hands[dealer])
            opening += 1
            #continue
            print(line1, end="")
            print(line2, end="")


    #print(opening, passing)
