import sys
import uuid
import shelve
import asyncio
import websockets
import functools

import game
import human

from nn.models import Models
from websockets.exceptions import ConnectionClosedError

MODELS = Models.load('../models')

print('models loaded')


def worker(driver):
    print('worker', driver)
    asyncio.new_event_loop().run_until_complete(driver.run())
board_no = []
boards = [...] 

async def handler(websocket, path, board_no):
    print(f'got websocket connection {websocket}')

    rdeal = tuple(boards[board_no[0]].replace("'","").rstrip('\n').split(','))

    print(f"Board: {board_no[0]+1}" )
    print(rdeal)

    driver = game.Driver(MODELS, human.WebsocketFactory(websocket))
    driver.set_deal(*rdeal)
    driver.human = [0.1, 0.1, 1, 0.1]

    try:
        await driver.run()

        with shelve.open('gamedb') as db:
            deal_bots = driver.to_dict()
            db[uuid.uuid4().hex] = deal_bots
            print('saved')
            board_no[0] = board_no[0] + 1
            if (board_no[0] > len(boards)):
                board_no[0] = 0
    
    except ConnectionClosedError  as ex:
        print('User left')
        

    except Exception as ex:
        print('Error:', ex)
        raise ex

with open("input.ben", "r") as file:
    board_no.append(0) 
    boards = file.readlines()

start_server = websockets.serve(functools.partial(handler, board_no=board_no), "0.0.0.0", 4443)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
