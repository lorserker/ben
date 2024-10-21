
# Function to load and list all entries from the pickle file, including averages and counts
from collections import Counter


def list_matching_entries():
    # Load data from the pickle file
    with open('GIB-Thorvald-8663-Optimum_OK.ben', 'r') as f:
        hands_data = f.readlines()
    
    with open('../data/Better bidding.txt', 'r') as f:
        sequences = f.readlines()

    print("Bid boards:",len(hands_data))
    print("Missing bidding sequences:",len(sequences))
    sequences = [sequence.replace('-', ' ').replace('\n', '') for sequence in sequences]

    print(sequences[0])

    key_counts = Counter()

    # Iterate through the list two lines at a time
    for i in range(0, len(hands_data), 2):
        # Read two lines at a time
        line1 = hands_data[i].strip()
        if i + 1 < len(hands_data):  # Ensure there's a second line
            line2 = hands_data[i + 1].strip()
        else:
            line2 = None  # Handle case where there's an odd number of lines

        key = ' '.join(line2.split()[2:]).replace("*","")
        if i == 101984:
            print(key)

        for sequence in sequences:  
            if key.startswith(sequence):
                #print(f"{line1} found for {key}")
                key_counts[sequence] += 1
    
    # Iterate through the Counter and print each key and count on a separate line
    for key, count in key_counts.items():
        print(f"{key}: {count}")
    
# Call the function to list entries
list_matching_entries()
