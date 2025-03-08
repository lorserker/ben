import random
import sys
import endplay.parsers.pbn as pbn

# Function to read the file and collect objects
def read_objects(filename):
    objects = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        lines = [line for line in lines if not line.strip().startswith('#')]
        for i in range(0, len(lines), 2):
            objects.append({
                'line1': lines[i].strip(),
                'line2': lines[i+1].strip()
            })
    return objects

# Function to select random objects
def select_random_objects(objects, num_objects):
    random.seed(42)
    return random.sample(objects, num_objects)

# Function to save selected objects to a new file
def save_objects_to_file(selected_objects, filename):
    with open(filename, 'w') as output_file:
        pbn.dump(selected_objects, output_file)

# Main function
def main():
    if len(sys.argv) < 2:
        print("Usage: python select_sample.py input_filename count")
        return
        
    # File containing 100000 lines
    input_filename = sys.argv[1]
    # Number of objects to select
    num_selected_objects = 1000
    if len(sys.argv) > 2:
        num_selected_objects = int(sys.argv[2])

    # New file to save selected objects
    output_filename = f'sampling_{num_selected_objects}.pbn'

    # Read the contents of the file
    with open(input_filename, 'r') as f:
        # Read objects from the input file
        boards = pbn.load(f)
    print(f"Found {len(boards)} deals")
    # Select random objects
    selected_objects = select_random_objects(boards, num_selected_objects)

    # Save selected objects to a new file
    save_objects_to_file(selected_objects, output_filename)

    print(f"{num_selected_objects} random objects saved to {output_filename}")

if __name__ == "__main__":
    main()


