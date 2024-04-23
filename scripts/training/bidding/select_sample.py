import random
import sys

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
    with open(filename, 'w') as file:
        for obj in selected_objects:
            file.write(obj['line1'] + '\n')
            file.write(obj['line2'] + '\n')

# Main function
def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_filename>")
        return
        
    # File containing 100000 lines
    input_filename = sys.argv[1]
    # Number of objects to select
    num_selected_objects = 1000
    # New file to save selected objects
    output_filename = f'sampling_{num_selected_objects}.ben'

    # Read objects from the input file
    objects = read_objects(input_filename)
    print(f"Found {len(objects)} deals")
    # Select random objects
    selected_objects = select_random_objects(objects, num_selected_objects)

    # Save selected objects to a new file
    save_objects_to_file(selected_objects, output_filename)

    print(f"{num_selected_objects} random objects saved to {output_filename}")

if __name__ == "__main__":
    main()
