from gevent import monkey
monkey.patch_all()

from bottle import Bottle, run, static_file, redirect

import shelve
import json
import os
import argparse

app = Bottle()
os.getcwd()

parser = argparse.ArgumentParser(description="Appserver")
parser.add_argument("--host", default="localhost", help="Hostname for appserver")
parser.add_argument("--port", type=int, default=8080, help="Port for appserver")
parser.add_argument("--db", default="gamedb", help="Db for appserver")

args = parser.parse_args()

port = args.port
DB_NAME = os.getcwd() + "/" + args.db
print("Reading deals from: "+DB_NAME)

script_dir = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
@app.route('/home')
def home():
    html = '<h1><a href="/app/bridge.html">Play Now</a></h1>\n'

    html += '<ul>\n'

    with shelve.open(DB_NAME) as db:
        deal_items = sorted(list(db.items()), key=lambda x: x[1]['timestamp'], reverse=True)
        for deal_id, deal in deal_items:
            board_no_ref = ""
            board_no_index = ""
            if 'board_number' in deal and deal['board_number'] is not None:
                board_no_ref = f"&board_number={deal['board_number']}"
                board_no_index = f"Board:{deal['board_number']}&nbsp;&nbsp;"
            html += '<li><span>{}<a href="/app/viz.html?deal={}{}">{} {}</a></span>&nbsp;&nbsp;&nbsp;'.format(board_no_index,deal_id, board_no_ref, deal['contract'], len(list(filter(lambda x: x % 2 == 1, deal['trick_winners']))))
            html += f'<span><a href="/api/delete/deal/{deal_id}">delete</a></span></li>\n'

    html += '</ul>'
    return html

@app.route('/app/<filename>')
def frontend(filename):
    if '?' in filename:
        filename = filename[:filename.index('?')]

    file_path = os.path.join(script_dir, 'frontend')    
    return static_file(filename, root=file_path)

@app.route('/api/deals/<deal_id>')
def deal_data(deal_id):
    db = shelve.open(DB_NAME)
    deal = db[deal_id]
    db.close()

    return json.dumps(deal)

@app.route('/api/delete/deal/<deal_id>')
def delete_deal(deal_id):
    db = shelve.open(DB_NAME)
    db.pop(deal_id)
    db.close()
    redirect('/home')

host = args.host
print(f'http://{host}:{port}/home')

run(app, host=host, port=port, server='gevent', log=None)
