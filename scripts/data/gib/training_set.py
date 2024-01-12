# Generate hands for GIB to bid and play
import time
import random

suits = ['S', 'H', 'D', 'C']
ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
dealer = ['N', 'E', 'S', 'W']
vuln = ['None', 'N-S', 'E-W', 'Both', 
        'N-S', 'E-W', 'Both', 'None',
        'E-W', 'Both', 'None', 'N-S', 
        'Both', 'None', 'N-S', 'E-W']
vulnGib = ['-', 'NS', 'EW', 'both', 
           'NS', 'EW', 'both', '-',
           'EW', 'both', '-', 'NS',
           'both', '-', 'NS', 'EW']


def sort_cards(hand):
    suit_order = {"S": 0, "H": 1, "D": 2, "C": 3}
    rank_order = {
        "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6, "8": 7, "9": 8,
        "T": 9, "J": 10, "Q": 11, "K": 12, "A": 13
    }

    hand.sort(key=lambda card: (
        suit_order[card[-1]], rank_order[card[:-1]]), reverse=True)


def shuffle(array):
    for i in range(len(array) - 1, 0, -1):
        j = random.randint(0, i)
        array[i], array[j] = array[j], array[i]


def distribute_cards(deck, num_cards):
    if num_cards <= 0 or num_cards > len(deck):
        raise ValueError("Invalid number of cards to distribute.")

    player_hand = []
    for _ in range(num_cards):
        player_hand.append(deck.pop())

    return player_hand


def create_deck():
    deck = []
    for suit in suits:
        for rank in ranks:
            deck.append(rank + suit)
    return deck


def print_hand(hand):
    formatted_hand = {suit: [] for suit in suits}

    for card in hand:
        suit = card[-1]
        formatted_hand[suit].append(card[:-1])

    printed_hand = ""
    for suit in formatted_hand:
        printed_hand += "".join(formatted_hand[suit]) + "."

    return printed_hand[:-1]  # Remove the trailing period at the end


def calculate_hcp(rank):
    hcp_values = {
        'A': 4,
        'K': 3,
        'Q': 2,
        'J': 1,
    }
    return hcp_values.get(rank, 0)


def format_card_count(card_count):
    return "-".join(f"{count:02}" for count in card_count.values())


def count_cards_in_hand(hand):
    card_count = {suit: 0 for suit in suits}

    for card in hand:
        card_count[card[-1]] += 1

    return card_count


def generate(boards):

    for i, board in enumerate(boards):
        deal_str = f"{print_hand(board[1]['north'])} {print_hand(board[1]['east'])} {print_hand(board[1]['south'])} {print_hand(board[1]['west'])}"

        yield {"dealer": dealer[i % 4], "vuln": vuln[i % len(vuln)], "deal_str": deal_str}


# Record the start time
start_time = time.time()

# Create a seeded random number generator
random.seed(0)

matched_hands = {}

for i in range(10000000):
    deck = create_deck()
    shuffle(deck)

    north = distribute_cards(deck, 13)
    east = distribute_cards(deck, 13)
    south = distribute_cards(deck, 13)
    west = distribute_cards(deck, 13)

    sort_cards(north)
    sort_cards(east)
    sort_cards(south)
    sort_cards(west)

    sorted_hcp = sum(calculate_hcp(card[0]) for card in north)

    card_count_string = format_card_count(count_cards_in_hand(north))

    key = f"{sorted_hcp:02}-{card_count_string}"
    matched_hands[key] = {"north": north, "east": east, "south": south, "west": west}
    
    if (i != 0) and i % 1000 == 0:
        print(f"Boards generated {i}. Boards selected: {len(matched_hands)}")
    if len(matched_hands) == 10000:
        print(f"Boards generated {i}. Boards selected: {len(matched_hands)}")
        break

matched_hands_array = list(matched_hands.items())
matched_hands_array.sort(key=lambda x: x[0], reverse=True)

output = ''
i = 0
for key, value in matched_hands_array:
    output += "-SWNE -T 30 -H" + "\n"
    output += print_hand(value["south"]) + "\n"
    output += print_hand(value["west"]) + "\n"
    output += print_hand(value["north"]) + "\n"
    output += print_hand(value["east"]) + "\n"
    output += dealer[i % 4] + " " + vulnGib[i % len(vuln)]  + "\n"
    i = i + 1
output += "-x" + "\n"


with open("input.gib", "w") as file:
    file.write(output)
    print('Output has been written to the file: input.gib')

output = ''
for deal_info in generate(matched_hands_array):
    output += f"{deal_info['dealer']} {deal_info['vuln']} {deal_info['deal_str']}\n"

output_file_path = "input.ben"
with open(output_file_path, "w") as file:
    file.write(output)
    print(f'Output has been written to the file: {output_file_path}')

print(len(matched_hands_array))

# Record the end time
end_time = time.time()

# Calculate the elapsed time in seconds
execution_time = end_time - start_time

# Display the result
print(f"Execution time: {execution_time:.2f} seconds")
