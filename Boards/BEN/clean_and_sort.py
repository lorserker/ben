import sys

# Read the file, sort lines, and remove duplicates
def read_and_sort_unique(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Remove duplicates by converting to a set, then sort
    unique_sorted_lines = sorted(set(line.strip() for line in lines))
    
    return unique_sorted_lines

# Usage: python clean_and_sort.py [deals.ben] > deals_cleaned.ben
filename = sys.argv[1] if len(sys.argv) > 1 else 'deals.ben'
sorted_unique_lines = read_and_sort_unique(filename)

# Print or save sorted unique lines
for line in sorted_unique_lines:
    print(line)
