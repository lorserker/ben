import sys
import uuid
import shelve
import asyncio
import websockets

import game
import human

from websockets.exceptions import ConnectionClosedError
from nn.models import Models
from urllib.parse import parse_qs, urlparse


MODELS = Models.load('../models')

print('models loaded')


def worker(driver):
    print('worker', driver)
    asyncio.new_event_loop().run_until_complete(driver.run())


async def handler(websocket, path):
    print(f'got websocket connection {websocket}')

    driver = game.Driver(MODELS, human.WebsocketFactory(websocket))

    parsed_url = urlparse(path)
    query_params = parse_qs(parsed_url.query)
    print(parsed_url)
    print(query_params)
    if query_params:
        P = query_params.get('P', [None])[0]
        print(f"P={P}")
        deal = query_params.get('deal', [None])[0]
        if deal:
            split_values = deal[1:-1].replace("'","").split(',')
            rdeal = tuple(value.strip() for value in split_values)
            driver.set_deal(*rdeal)
        if P == "0":
            driver.human = [0.1, -1, 0.1, -1]
        if P == "1":
            driver.human = [-1, 0.1, -1, 0.1]
        if P == "2":
            driver.human = [0.05, -1, 0.05, -1]
        if P == "3":
            driver.human = [-1, 0.05, -1, 0.05]
    else:
        #Just take a random"
        rdeal = game.random_deal()
        # example of to use a fixed deal
        #rdeal = ('KJ876.5.97542.J4 2.AQT4.AT8.K8652 AQ9543.KJ76.Q.Q9 T.9832.KJ63.AT73', 'S None')
        driver.human = [0.1, 0.1, 1, 0.1]
        driver.set_deal(*rdeal)

    try:
        await driver.run()

        with shelve.open('gamedb') as db:
            deal_bots = driver.to_dict()
            db[uuid.uuid4().hex] = deal_bots
            print('saved')
    
    except ConnectionClosedError  as ex:
        print('User left')
        
    except Exception as ex:
        print('Error:', ex)
        raise ex


start_server = websockets.serve(handler, "0.0.0.0", 4443)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
