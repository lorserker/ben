import json

from bottle import Bottle, request, run
from gevent import monkey

from bots import BotBid
from nn.models import Models

MODELS = Models.load('../models')

monkey.patch_all()

app = Bottle()


@app.route('/api/bid', method='POST')
def deal_data():
    req = request.json
    print("got request", req)

    bot_bid = BotBid(req['vul'], req['hand'], MODELS)
    bid = bot_bid.bid(req['auction']).bid

    print("bid response", bid)
    return json.dumps({"bid": bid})


run(app, host='0.0.0.0', port=8080, server='gevent')
