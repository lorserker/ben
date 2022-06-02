import sys
import uuid
import shelve
import asyncio
import websockets

import game
import human

from nn.models import Models


MODELS = Models.load('../models')

print('models loaded')


def worker(driver):
    print('worker', driver)
    asyncio.new_event_loop().run_until_complete(driver.run())


async def handler(websocket, path):
    print(f'got websocket connection {websocket}')

    rdeal = game.random_deal()
    
    driver = game.Driver(MODELS, human.WebsocketFactory(websocket))
    driver.set_deal(*rdeal)
    driver.human = [False, False, True, False]

    try:
        await driver.run()

        with shelve.open('gamedb') as db:
            deal_bots = driver.to_dict()
            db[uuid.uuid4().hex] = deal_bots
            print('saved')
    
    except Exception as ex:
        print('Error:', ex)
        raise ex


start_server = websockets.serve(handler, "0.0.0.0", 443)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
