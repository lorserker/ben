import subprocess
import tempfile
import os
import deck52
import re
import time

def alphamju(decisions, tricks, suit, cards, worlds, rng, verbose):
    # Define the executable and parameters
    exe_path = "alphamju/alphamju.exe"
    # Always minimum 2 decisions and due to timer no more than 4
    decisions = min(max(decisions,2),4)
    loglevel = 1
    worlds = [f'[Deal "W:{deal}"]' for deal in worlds]    

    if len(worlds) > 100:
        worlds = list(rng.choice(worlds, 100, replace=False))  # Unique random selection

    #print("Worlds: ", worlds)

    # AlphaMju has always South as declarer, so we need to find who did lead this trick
    player_str = "SENW"[(len(cards)) % 4]

    # Create a temporary file and write the worlds data to it
    with tempfile.NamedTemporaryFile(delete=False, mode='w', newline='') as temp_file:
        temp_file.write('\n'.join(worlds))
        temp_file_path = temp_file.name

    params = [str(decisions), str(loglevel), str(tricks), suit, temp_file_path, player_str]

    if cards:
        for c in cards:
            params.append(deck52.decode_card(c))
    #if verbose:
    print("Params: ", params)
    start_time = time.time()  # Start timing
    try:
        # Run the process and capture the output
        result = subprocess.run([exe_path] + params, capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        print("Params: ", params)
        print("Process timed out after 10 seconds")
        return []
    finally:
        end_time = time.time()  # End timing
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.3f} seconds")

    res = []
    # Match 2-character card code (suit + rank), whitespace, then a number
    pattern = r':([SHDC][AKQJT98765432])\s+(\d+)\b'

    print("AlphaMjuResult: ", result.stdout)
    matches = re.findall(pattern, result.stdout)

    # Get up to 13 matches
    for card, freq in matches:
        #print(f"{card}    {freq}")
        res.append((card, int(freq)))

    # Sort by percentage in descending order
    res.sort(key=lambda x: x[1], reverse=True)
    print("AlphaMjuResult: ", res)
    if result.stderr:
        print("Error: ", result.stderr)
    #print(res)
    return res

