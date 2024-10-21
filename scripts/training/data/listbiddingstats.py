import pickle

# Define the pickle DB path
pickle_db_path = 'hands_data.pkl'

# Function to load and list all entries from the pickle file, including averages and counts
def list_pickle_entries(pickle_db_path):
    try:
        # Load data from the pickle file
        with open(pickle_db_path, 'rb') as pickle_file:
            hands_data = pickle.load(pickle_file)
        
        # Iterate and print all entries
        for key, hands in hands_data.items():
            if "N-S P P 1C 1D X 2C" in key:
                print(f"Key: {key}")
                for hand_key, stats in hands.items():
                    print(f"  {hand_key}:")
                    for suit, values in stats.items():
                        min_value, max_value, total_sum, total_count = values
                        average = total_sum / total_count if total_count != 0 else 0
                        if suit != 'HCP':
                            print(f"    {suit}: Min = {min_value}, Max = {max_value}, Average = {average:.2f}, Count = {total_count}")
                        else:
                            print(f"    Honor Points: Min = {min_value}, Max = {max_value}, Average = {average:.2f}, Count = {total_count}")
                print()  # Print a blank line between entries
    except FileNotFoundError:
        print(f"Error: File {pickle_db_path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Call the function to list entries
list_pickle_entries(pickle_db_path)
