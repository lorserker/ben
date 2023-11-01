def is_valid_bid(line):
    # Split the line into tokens
    tokens = line.split()

    # Ignore the first two tokens
    tokens = tokens[2:]

    # Remove any "P" tokens
    tokens = [token for token in tokens if token != 'P']

    # Check if the first bid (after removing "P" tokens) is "2N"
    if tokens[0] == '4C':
        return True
    else:
        return False

# Define a dictionary to map directions to index updates
direction_to_index = {'N': 0, 'E': 1, 'S': 2, 'W': 3}

def get_hcp(hand):
    hcp = {'A': 4, 'K': 3, 'Q': 2, 'J': 1}
    return sum([hcp.get(c, 0) for c in hand])

# Read the file and validate bid for each board
with open('../Jack/BW5C_Total.ben', 'r') as file:
    lines = list(file)
    for board_number, (line1, line2) in enumerate(zip(lines[::2], lines[1::2]), start=1):
        bidding = [{'bid': bid} for bid in line2.split()[2:]]
        for bid_dict in bidding:
            if bid_dict['bid'] == 'P':
                bid_dict['bid'] = 'PASS'

        if (is_valid_bid(line2)):
            hands = line1.split()
            bids = line2.split()
            print(bids)

            # Find the index of "2N" bid in the bids list based on the direction
            bid_index = bids.index('1N') + direction_to_index.get(bids[0], 0) if '1N' in bids else -1
            # Find the index of the "2N" bid in the bids list
            bid_index = (bid_index - 2) % 4
            #if bid_index != -1 and bid_index < len(hands):
                #print(f"Bid: 2N, Hand: {hands[bid_index]} {get_hcp(hands[bid_index])}")
            
            print(line1,end="")
            print(line2,end="")