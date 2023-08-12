import uuid
import shelve
import asyncio
import websockets
import argparse
import game
import human
import conf
import functools
import os

from websockets.exceptions import ConnectionClosedError
from nn.models import Models
from sample import Sample
from urllib.parse import parse_qs, urlparse
from pbn2ben import load


random = True
#For some strange reason parameters parsed to the handler must be an array
board_no = []
board_no.append(0) 


parser = argparse.ArgumentParser(description="Game server")
parser.add_argument("--boards", default="", help="Filename for configuration")
parser.add_argument("--boardno", default=0, type=int, help="Board number to start from")
parser.add_argument("--config", default="./config/default.conf", help="Filename for configuration")
parser.add_argument("--ns", type=int, default=-1, help="System for NS")
parser.add_argument("--ew", type=int, default=-1, help="System for EW")
args = parser.parse_args()

configfile = args.config

if args.boards:
    filename = args.boards
    file_extension = os.path.splitext(filename)[1].lower()  
    if file_extension == '.ben':
        with open(filename, "r") as file:
            board_no.append(0) 
            boards = file.readlines()
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
    board_no[0] = args.boardno

if random:
    print("Playing random deals or deals from the client")

ns = args.ns
ew = args.ew


models = Models.from_conf(conf.load(configfile))

print('models loaded')


def worker(driver):
    print('worker', driver)
    asyncio.new_event_loop().run_until_complete(driver.run())


async def handler(websocket, path, board_no):
    print(f'Got websocket connection {websocket}')

    driver = game.Driver(models, human.WebsocketFactory(websocket), Sample.from_conf(conf.load(configfile)))

    parsed_url = urlparse(path)
    query_params = parse_qs(parsed_url.query)

    if query_params:
        P = query_params.get('P', [None])[0]
        deal = query_params.get('deal', [None])[0]
        if deal:
            split_values = deal[1:-1].replace("'","").split(',')
            rdeal = tuple(value.strip() for value in split_values)
            driver.set_deal(*rdeal, ns, ew)
        if P == "0":
            driver.human = [0.1, -1, 0.1, -1]
        if P == "1":
            driver.human = [-1, 0.1, -1, 0.1]
        if P == "2":
            driver.human = [0.05, -1, 0.05, -1]
        if P == "3":
            driver.human = [-1, 0.05, -1, 0.05]
        if P == "4":
            driver.human = [0.1, 0.1, 0.1, 0.1]
        if P == "5":
            driver.human = [0.1, 0.1, 1, 0.1]

    else:
        if random:
            #Just take a random"
            rdeal = game.random_deal()
            # example of to use a fixed deal
            # rdeal = ('T7654.A.JT54.AK8 Q2.JT9.AK32.Q754 KJ983.6.Q9.JT932 A.KQ875432.876.6', 'E E-W', -1, -1)
            driver.human = [0.1, 0.1, 1, 0.1]
            driver.set_deal(*rdeal, ns, ew)
        else:
            rdeal = tuple(boards[board_no[0]].replace("'","").rstrip('\n').split(','))
            print(f"Board: {board_no[0]+1}" )
            print(rdeal)
            driver.set_deal(*rdeal, ns, ew)
            driver.human = [0.1, 0.1, 1, 0.1]

    try:
        await driver.run()

        with shelve.open('gamedb') as db:
            deal_bots = driver.to_dict()
            db[uuid.uuid4().hex] = deal_bots
            print('saved')
            if not random:
                board_no[0] = board_no[0] + 1
                if (board_no[0] > len(boards)):
                    board_no[0] = 0

    
    except ConnectionClosedError  as ex:
        print('User left')
        
    except Exception as ex:
        print('Error:', ex)
        raise ex


start_server = websockets.serve(functools.partial(handler, board_no=board_no), "0.0.0.0", 4443)


asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
