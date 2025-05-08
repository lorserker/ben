import os

# Directory containing the files
directory_path = './'

# Regular expression pattern to match text enclosed in backticks spanning multiple lines
pattern = r'`(.*?)`'

# Function to find and process text enclosed in backticks and save to a .dlr file


def rotate_deal(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        processed_text = rotate_hand(content)
        output_file_path = os.path.splitext(file_path)[0]+ "-rotated" + ".pbn"
        with open(output_file_path, 'w') as output_file:
            # Save the processed text to the .dlr file
            print(output_file.name, "rotated")
            output_file.write(processed_text)


def rotate_hand(extracted_text):
    processed_text = []
    lines = extracted_text.split('\n')
    prefix_mapping = {"N:": "E:", "E:": "S:", "S:": "W:", "W:": "N:"}
    processed_text = []

    for line in lines:
        if line.startswith("[Deal "):
            prefix = line[7:9]  # Extract the prefix
            replacement = prefix_mapping.get(prefix, None)
            if replacement:
                processed_text.append(line.replace(prefix, replacement, 1))
            else:
                processed_text.append(line)
        else:
            processed_text.append(line)
    return '\n'.join(processed_text)


# List all files in the directory
for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    # Check if it's a file
    if os.path.isfile(file_path) and (filename.endswith('.pbn') and (filename.find('-rotated') == -1)):
        rotate_deal(file_path)
