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

# Define a dictionary to map directions to index updates
direction_to_index = {'N': 0, 'E': 1, 'S': 2, 'W': 3}

# Read the file and validate bid for each board
with open('GIB-Thorvald-2024-08-09.ben', 'r') as file:
    lines = list(file)
    for board_number, (line1, line2) in enumerate(zip(lines[::2], lines[1::2]), start=1):
        hands = line1.split()
        bids = line2.split()
        if "P" not in bids[2]:
            dealer = direction_to_index[bids[0]]
            if dealer == 0:
                print(line1, end="")
                print(line2, end="")
