import pickle

# Define file path and pickle DB path
file_path = 'GIB-Thorvald-8663-optimum_OK.ben'
pickle_db_path = 'hands_data.pkl'

# Suit mapping: Indexes correspond to ['S', 'H', 'D', 'C']
suit_order = ['S', 'H', 'D', 'C']

# Initialize the main dictionary to hold condensed data
hands_data = {}

# Helper function to create default min-max-sum-count dict for a hand (including honor points)
def default_hand_stats():
    return {'S': [13, 0, 0, 0], 'H': [13, 0, 0, 0], 'D': [13, 0, 0, 0], 'C': [13, 0, 0, 0], 'HCP': [40, 0, 0, 0]}

# Helper function to calculate honor points for a hand (4 for A, 3 for K, 2 for Q, 1 for J)
def calculate_hcp(hand):
    points = 0
    points += hand.count('A') * 4
    points += hand.count('K') * 3
    points += hand.count('Q') * 2
    points += hand.count('J') * 1
    return points

# Helper function to update min and max counts for each suit across all hands for a key
def update_global_hand_stats(current_hand_stats, hand):
    # Split hand into suits by '.'
    suits = hand.split('.')
    
    # Ensure hand has exactly 4 suits
    if len(suits) != 4:
        print(f"Error: Expected 4 suits but found {len(suits)} in hand: {hand}")
        return
    
    # Count number of cards per suit
    suit_counts = [len(s) for s in suits]
    
    # Calculate honor points for the hand
    total_hcp = sum(calculate_hcp(suit) for suit in suits)
    
   # Update the global min, max, sum, and count for each suit and honor points
    for i, count in enumerate(suit_counts):
        current_min, current_max, current_sum, current_count = current_hand_stats[suit_order[i]]
        current_hand_stats[suit_order[i]] = [
            min(current_min, count),  # Update min
            max(current_max, count),  # Update max
            current_sum + count,      # Update sum
            current_count + 1         # Increment count
        ]
    
    # Update the global min, max, sum, and count for honor points
    current_honor_min, current_honor_max, current_honor_sum, current_honor_count = current_hand_stats['HCP']
    current_hand_stats['HCP'] = [
        min(current_honor_min, total_hcp),   # Update min
        max(current_honor_max, total_hcp),   # Update max
        current_honor_sum + total_hcp,       # Update sum
        current_honor_count + 1                       # Increment count
    ]

# Read file and process hands
with open(file_path, 'r') as file:
    hands_line = None  # Track the last read hands line
    while True:
        line = file.readline().strip()
        
        # If the line is empty, end of file
        if not line:
            break
        
        # Ignore lines starting with '#'
        if line.startswith('#'):
            continue
        
        # Alternate between hands_line and key_line
        if hands_line is None:
            # First valid line is the hands line
            hands_line = line
        else:
            # Second valid line is the key line
            key_line = line
        
            # Parse hands (4 hands delimited by space)
            hands = hands_line.split()

    # Ensure exactly 4 hands are present
            if len(hands) != 4:
                print(hands)
                print(f"Error: Expected 4 hands but found {len(hands)} for key {key_line}")
                continue  # Skip this record
            
            bids = key_line.split()[2:]
            new_key = ' '.join(key_line.split()[:2])
            for i in range(len(bids)):
                # This should loop over the entire auction
                # Initialize the entry for the key if it doesn't exist
                new_key += ' ' + bids[i] 
                if new_key not in hands_data:
                    hands_data[new_key] = {
                        'hand1': default_hand_stats(),
                        'hand2': default_hand_stats(),
                        'hand3': default_hand_stats(),
                        'hand4': default_hand_stats()
                    }

                if new_key == "N None 1N P 3C*":
                    print(hands)
                    #hand = hand[2]
                    # Split hand into suits by '.'
                    #suits = hand.split('.')                
                    # Count number of cards per suit
                    #suit_counts = [len(s) for s in suits]
                    # Calculate honor points for the hand
                    #total_hcp = sum(calculate_hcp(suit) for suit in suits)
                    #if total_hcp > 9:
                    #    print(hand, total_hcp)

                # For each hand (there are 4 hands), update statistics in hands_data using the key
                for i, hand in enumerate(hands):
                    hand_key = f'hand{i+1}'
                    update_global_hand_stats(hands_data[new_key][hand_key], hand)
 
 # Reset hands_line for the next pair
            hands_line = None
#
#  Save the result to the pickle database
with open(pickle_db_path, 'wb') as pickle_file:
    pickle.dump(hands_data, pickle_file)

print(f"Data has been condensed and stored in {pickle_db_path}")

