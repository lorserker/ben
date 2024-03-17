import sys
import endplay.parsers.pbn as pbn
from endplay.types.board import Board
import endplay.config as config
import argparse
import io
import os
import math
import bisect
import tkinter as tk
from tkinter import filedialog


IMP = [10, 40, 80, 120, 160, 210, 260, 310, 360, 420, 490, 590, 740, 890, 1090, 1290, 
       1490, 1740, 1990, 2240, 2490, 3490, 3990]


def get_imps(score1, score2):
    score_diff = score1 - score2
    imp = bisect.bisect_left(IMP, int(math.fabs(score_diff)))
    if score_diff >= 0:
        return imp
    else:
        return -imp


# Define a function to convert room values to numeric values for sorting
def room_to_numeric(room):
    if room == "Open":
        return 0
    elif room == "Closed":
        return 1
    else:
        return 2  # Handle other cases as needed

def remove_feasability_lines(file_path):
    # Read the contents of the file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Filter out lines starting with '{Feasability:'
    new_lines = [line for line in lines if not line.startswith('{Feasability:')]

    # Create an in-memory file-like object
    fake_file = io.StringIO(''.join(new_lines))
    
    return fake_file

def write_playernames_to_file(output_file, sorted_boards):
    output_file.write("pn|")
    output_file.write(f"{sorted_boards[0].info.south},")
    output_file.write(f"{sorted_boards[0].info.west},")
    output_file.write(f"{sorted_boards[0].info.north},")
    output_file.write(f"{sorted_boards[0].info.east},")
    output_file.write(f"{sorted_boards[1].info.south},")
    output_file.write(f"{sorted_boards[1].info.west},")
    output_file.write(f"{sorted_boards[1].info.north},")
    output_file.write(f"{sorted_boards[1].info.east}\n")
    output_file.write("|pg||\n")


def main():

    print("Table Manager PBN cleaner, Version 1.0.8")

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Example program with command-line arguments")

    # Add a positional argument for the name
    parser.add_argument("filename", help="filename for conversion")

    # Parse the command-line arguments
    args = parser.parse_args()

    if args.filename == None:

        # create a root window
        root = tk.Tk()
        root.withdraw()

        # specify the allowed file types
        file_types = [
            ("PBN files", "*.pbn"),  # Example: Only allow .txt files
            ("All files", "*.*")     # Allow all files (in case the user wants to choose other formats)
        ]
        # open the file dialog box
        file_path = filedialog.askopenfilename(initialdir=".", filetypes=file_types)

        # print the selected file path
        if not file_path:
            sys.exit(1)
    else:
        file_path = args.filename

    # Call the function to remove lines and save the file
    fakefile = remove_feasability_lines(file_path)

    try:
        boards = pbn.load(fakefile)
    except pbn.PBNDecodeError as e:
        print("OK: There was an issue decoding the PBN file.")
        print("Error message:", e)
        print("Current line:", e.line)
        print("Line number:", e.lineno)
        sys.exit(1)

    #Don't use unicode
    config.use_unicode = False 

    # Construct the output file
    # Split the file path into directory and filename
    directory, filename = os.path.split(file_path)

    # Insert "DDS" at the beginning of the filename
    new_filename = "Cleaned_" + filename

    if args.filename == None:
        # Get the file path to save the data
        output_file = filedialog.asksaveasfile(defaultextension=".pbn", initialdir=directory, filetypes=file_types, initialfile=new_filename)
    else:
        output_file = open(new_filename, 'w')

    if output_file is not None:
        selected_attributes = ['board_num', 'dealer', 'vul', 'deal']

        new_boards = []

        for index, board in enumerate(boards, start = 1):
            attributes = {attr: getattr(board, attr) for attr in selected_attributes}
            attributes['info'] = Board.Info()  # Add an empty 'info' dictionary
            new_board = Board(**attributes)  # Create a new Board object with selected attributes
            new_board.board_num = index  # Set the 'board_num' attribute explicitly
            if board.info.room != "Closed":
                new_boards.append(new_board)

        pbn.dump(new_boards, output_file)
        # Close the file after writing
        output_file.close()
        print(f"{output_file.name} generated")
    else:
        print("File not saved")


if __name__ == "__main__":
    main()

