import subprocess
import tempfile
import os
import deck52

def alphamju(tricks, suit, player, cards, worlds):
    # Define the executable and parameters
    exe_path = "alphamju/alphamju.exe"
    decisions = 2
    loglevel = 3
    worlds = [f'[Deal "W:{deal}"]' for deal in worlds]    

    print("Worlds: ", worlds)

    player = "NESW"[player]
        # Create a temporary file and write the worlds data to it
    with tempfile.NamedTemporaryFile(delete=False, mode='w', newline='') as temp_file:
        temp_file.write('\n'.join(worlds))
        temp_file_path = temp_file.name

    cards = " ".join(deck52.decode_card(c) for c in cards)
    params = [str(decisions), str(loglevel), str(tricks), suit, temp_file_path, player]

    if cards:
        params.append(cards)

    print("Params: ", params)

    # Run the process and capture the output
    result = subprocess.run([exe_path] + params, capture_output=True, text=True)

    print("Result: ", result.stdout)
    print("Error: ", result.stderr)
    res = []
    # Split the text into lines and check for the '<' character
    for line in result.stdout.splitlines():
        if '<' in line:
            res.append(line)
    #os.remove(temp_file_path)
    return res

