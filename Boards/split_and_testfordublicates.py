def write_data_to_files(lines_seen, base_filename='input'):
    file_count = 1
    lines_written = 0
    max_lines_per_file = 200000
    current_filename = f"{base_filename}_{file_count}.bba"
    print(f"Writing to {current_filename}")
    file = open(current_filename, 'w')

    for line in lines_seen:
        if lines_written >= max_lines_per_file:
            file.close()
            file_count += 1
            current_filename = f"{base_filename}_{file_count}.bba"
            print(f"Writing to {current_filename}")
            file = open(current_filename, 'w')
            lines_written = 0

        file.write(line + '\n')
        lines_written += 1

    file.close()

def find_duplicate_lines(filename):
    lines_seen = set()  # Set to store unique lines
    duplicate_lines = []  # List to store duplicate lines

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace
            if line in lines_seen:
                print(f"Duplicate line found: {line}")
                duplicate_lines.append(line)
            else:
                lines_seen.add(line)

    print(len(lines_seen), "unique lines found")
    write_data_to_files(lines_seen)
    return duplicate_lines

# Example usage:
filename = 'input.bba'  # Replace 'your_file.txt' with the path to your file
duplicates = find_duplicate_lines(filename)
print("Duplicate lines found:")
for line in duplicates:
    print(line)
print("and removed")

