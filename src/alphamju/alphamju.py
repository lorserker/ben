import subprocess
import tempfile
import os
import deck52
import re
import time

def alphamju(tricks, suit, player, cards, worlds, rng):
    # Define the executable and parameters
    exe_path = "alphamju/alphamju.exe"
    # Always minimum 2 decisions
    decisions = max(tricks - 2,2)
    loglevel = 1
    worlds = [f'[Deal "W:{deal}"]' for deal in worlds]    

    if len(worlds) > 100:
        worlds = list(rng.choice(worlds, 100, replace=False))  # Unique random selection

    #print("Worlds: ", worlds)

    # AlphaMju has always South as declarer, so we need to find who did lead this trick
    #if player == 1:
    #    player_str = "NWSE"[(len(cards)) % 4]
    #if player == 3:
    player_str = "SENW"[(len(cards)) % 4]

    # Create a temporary file and write the worlds data to it
    with tempfile.NamedTemporaryFile(delete=False, mode='w', newline='') as temp_file:
        temp_file.write('\n'.join(worlds))
        temp_file_path = temp_file.name

    params = [str(decisions), str(loglevel), str(tricks), suit, temp_file_path, player_str]

    if cards:
        for c in cards:
            params.append(deck52.decode_card(c))

    print("Params: ", params)
    start_time = time.time()  # Start timing
    try:
        # Run the process and capture the output
        result = subprocess.run([exe_path] + params, capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        print("Process timed out after 10 seconds")
        return []
    finally:
        end_time = time.time()  # End timing
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.3f} seconds")

    #user_input = input("\n Q to quit or any other key for next deal.\n ")


    player_str = "S"

    res = []
    for line in result.stdout.split('\n'):
        #print(player,player_str,line)
        if f"{player_str}:" in line:
            parts = line.split(f"{player_str}:")[1]  # Extract everything after "N:"
            card = parts[:2].strip()        # First two characters are the card
            percentage = parts[2:6].strip()  # Next three characters (right-aligned number) + '%' 
            numeric_value = int(percentage.replace('%', ''))  # Convert to int for sorting
            res.append((card, numeric_value))

    # Sort by percentage in descending order
    res.sort(key=lambda x: x[1], reverse=True)

    if result.stderr:
        print("Error: ", result.stderr)
    print(res)
    #user_input = input("\n Q to quit or any other key for next deal.\n ")
    return res

