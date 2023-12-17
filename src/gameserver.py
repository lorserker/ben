import os
import logging

# Set logging level to suppress warnings
logging.getLogger().setLevel(logging.ERROR)
# Just disables the warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# This import is only to help PyInstaller when generating the executables
import tensorflow as tf

import uuid
import shelve
import time
import asyncio
import websockets
import argparse
import game
import human
import conf
import functools
import os
import numpy as np
from websockets.exceptions import ConnectionClosedOK
from sample import Sample
from urllib.parse import parse_qs, urlparse
from pbn2ben import load

def get_execution_path():
    # Get the directory where the program is started from either PyInstaller executable or the script
    return os.getcwd()

random = True
#For some strange reason parameters parsed to the handler must be an array
board_no = []
board_no.append(0) 

# Get the path to the config file
config_path = get_execution_path()
    
base_path = os.getenv('BEN_HOME') or config_path

parser = argparse.ArgumentParser(description="Game server")
parser.add_argument("--boards", default="", help="Filename for boards")
parser.add_argument("--boardno", default=0, type=int, help="Board number to start from")
parser.add_argument("--config", default=f"{base_path}/config/default.conf", help="Filename for configuration")
parser.add_argument("--verbose", type=bool, default=False, help="Output samples and other information during play")
parser.add_argument("--port", type=int, default=4443, help="Port for appserver")
parser.add_argument("--auto", type=bool, default=False, help="BEN bids and plays all 4 hands")
parser.add_argument("--playonly", type=bool, default=False, help="Only play, no bidding")

args = parser.parse_args()

configfile = args.config
verbose = args.verbose
port = args.port
auto = args.auto
play_only = args.playonly

if args.boards:
    filename = args.boards
    file_extension = os.path.splitext(filename)[1].lower()  
    if file_extension == '.ben':
        with open(filename, "r") as file:
            board_no.append(0) 
            boards = []
            lines = file.readlines()  # 
            # Loop through the lines, grouping them into objects
            for i in range(0, len(lines), 2):
                board = {
                    'deal': lines[i].strip(),      
                    'auction': lines[i+1].strip().replace('NT','N')  
                }
                boards.append(board)            
            print(f"{len(boards)} boards loaded from file")
        random = False
    if file_extension == '.pbn':
        with open(filename, "r") as file:
            lines = file.readlines()
            boards = load(lines)
            print(f"{len(boards)} boards loaded from file")
        random = False

if args.boardno:
    print(f"Starting from {args.boardno}")
    board_no[0] = args.boardno -1

if random:
    print("Playing random deals or deals from the client")

np.set_printoptions(precision=2, suppress=True, linewidth=240)

configuration = conf.load(configfile)

try:
    if (configuration["models"]['tf_version'] == "2"):
        print("Loading version 2")
        from nn.models_tf2 import Models
    else: 
        # Default to version 1. of Tensorflow
        from nn.models import Models
except KeyError:
        # Default to version 1. of Tensorflow
        from nn.models import Models

models = Models.from_conf(configuration, base_path.replace(os.path.sep + "src",""))

print('models loaded')


def worker(driver):
    print('worker', driver)
    asyncio.new_event_loop().run_until_complete(driver.run())


async def handler(websocket, path, board_no):
    print("Got websocket connection")

    driver = game.Driver(models, human.WebsocketFactory(websocket), Sample.from_conf(configuration, verbose), verbose)
    play_only = False
    if (auto):
        driver.human = [False, False, False, False]
    parsed_url = urlparse(path)
    query_params = parse_qs(parsed_url.query)

    if query_params:
        P = query_params.get('P', [None])[0]
        if P == "5":
            play_only = True
        deal = query_params.get('deal', [None])[0]
        board_no_query = query_params.get('board_no', [None])[0]
        if deal:
            split_values = deal[1:-1].replace("'","").split(',')
            rdeal = tuple(value.strip() for value in split_values)
            driver.set_deal(board_no_query, *rdeal, play_only)
            print(f"Board: {board_no_query} {rdeal}")
    else:
        if random:
            #Just take a random"
            rdeal = game.random_deal()
            # example of to use a fixed deal
            # rdeal = ('AK64.8642.Q32.Q6 9.QT973.AT5.KJ94 QT532.J5.KJ974.7 J87.AK.86.AT8532', 'W None')
            driver.set_deal(None, *rdeal, False)
            print(f"Deal: {rdeal}")
        else:
            rdeal = boards[board_no[0]]['deal']
            auction = boards[board_no[0]]['auction']
            print(f"Board: {board_no[0]+1} {rdeal}")
            driver.set_deal(board_no[0] + 1, rdeal, auction, play_only)

    try:
        t_start = time.time()
        await driver.run()

        with shelve.open(f"{get_execution_path()}/gamedb") as db:
            deal_bots = driver.to_dict()
            print("Saving Board: ",driver.hands)
            print(f'Board played in {time.time() - t_start:0.1f} seconds')
            db[uuid.uuid4().hex] = deal_bots
            print('Deal saved')
            if not random:
                board_no[0] = (board_no[0] + 1) % len(boards)

    
    except ConnectionClosedOK  as ex:
        print('User left')
        
    except Exception as ex:
        print('Error:', ex)
        raise ex

async def main():
    print("Listening on port: ",port)
    start_server = websockets.serve(functools.partial(handler, board_no=board_no), "0.0.0.0", 4443)
    await start_server

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()

