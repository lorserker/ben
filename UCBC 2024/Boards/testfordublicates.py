def find_duplicate_lines(filename):
    lines_seen = set()  # Set to store unique lines
    duplicate_lines = []  # List to store duplicate lines

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace
            if line in lines_seen:
                duplicate_lines.append(line)
            else:
                lines_seen.add(line)

    return duplicate_lines

# Example usage:
filename = 'input.bba'  # Replace 'your_file.txt' with the path to your file
duplicates = find_duplicate_lines(filename)
print("Duplicate lines found:")
for line in duplicates:
    print(line)
