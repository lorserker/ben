from gevent import monkey

from bots import BotBid
monkey.patch_all()

from bottle import Bottle, run, static_file, redirect, template, request, response

import bottle
import json
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
import datetime
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
# For some strange reason parameters parsed to the handler must be an array
board_no = []
seed = None
board_no.append(0) 

# Get the path to the config file
config_path = get_execution_path()
    
base_path = os.getenv('BEN_HOME') or config_path

parser = argparse.ArgumentParser(description="Game API")
parser.add_argument("--host", default="localhost", help="Hostname for appserver")
parser.add_argument("--config", default=f"{base_path}/config/default.conf", help="Filename for configuration")
parser.add_argument("--verbose", type=bool, default=False, help="Output samples and other information during play")
parser.add_argument("--port", type=int, default=8085, help="Port for appserver")

args = parser.parse_args()

configfile = args.config
verbose = args.verbose
port = args.port
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
sampler = Sample.from_conf(configuration, verbose)

print('models loaded')

host = args.host
print(f'http://{host}:{port}/home')

app = Bottle()

# CORS middleware
class CorsPlugin(object):
    name = 'cors'
    api = 2

    def apply(self, callback, route):
        def wrapper(*args, **kwargs):
            response.headers['Access-Control-Allow-Origin'] = '*'  # Replace * with your allowed domain if needed
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'
            if request.method != 'OPTIONS':
                return callback(*args, **kwargs)
        return wrapper

app.install(CorsPlugin())

@app.route('/')
def default():
    html = '<h1><a href="/app/bridge.html?S=x&A=2&T=2">Play Now</a></h1>\n'
    return html

@app.route('/hint')
def frontend():
    # First we extract our hand
    hand = request.query.get("hand").replace('_','.')[:-1]
    print(hand)
    # Then vulnerability
    v = request.query.get("vul")
    print(v)
    vuln = []
    vuln.append('@v' in v)
    vuln.append('@V' in v)
    # And finally the bidding, where we deduct dealer and our position
    dealer_i = 0
    position = 0
    ctx = request.query.get("ctx")
    # Split the string into chunks of every second character
    bids = [ctx[i:i+2] for i in range(0, len(ctx), 2)]
    print(bids)
    auction = [bid.replace('--', "PASS").replace('Db', 'X').replace('Rd', 'XX') for bid in bids]
    hint_bot = BotBid(vuln, hand, models, sampler, position, dealer_i, verbose)
    bid = hint_bot.bid(auction)
    print(bid.bid)
    return json.dumps(bid.to_dict())

if __name__ == "__main__":
    run(app, host=host, port=port, server='gevent', log=None)

