from gevent import monkey
monkey.patch_all()

from bottle import Bottle, run, static_file, redirect, template, request, response

import bottle
bottle.BaseRequest.MEMFILE_MAX = 5 * 1024 * 1024 
import shelve
import json
import os
import argparse
import uuid

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
@app.route('/play')
@app.route('/home')
def home():
    html = '<h1><a href="/app/bridge.html?S=x&A=2&T=2&name=Human">Play Now</a></h1>\n'

    html += '<ul>\n'

    with shelve.open(DB_NAME) as db:
        deal_items = sorted(list(db.items()), key=lambda x: x[1]['timestamp'], reverse=True)
        for deal_id, deal in deal_items:
            board_no_ref = ""
            board_no_index = ""
            if 'board_number' in deal and deal['board_number'] is not None:
                board_no_ref = f"&board_number={deal['board_number']}"
                board_no_index = f"Board:{deal['board_number']}&nbsp;&nbsp;"
            tricks = len(list(filter(lambda x: x % 2 == 1, deal['trick_winners'])))
            if 'claimed' in deal:
                if 'claimedbydeclarer' in deal:
                    if deal['claimedbydeclarer']:
                        tricks += deal['claimed']
                    else:
                        tricks += (13-tricks-deal['claimed'])

            if deal['contract'] is not None:
                html += '<li><span>{}<a href="/app/viz.html?deal={}{}">{} {}</a></span>&nbsp;&nbsp;&nbsp;'.format(board_no_index,deal_id, board_no_ref, deal['contract'], tricks)
            else:
                html += '<li><span>{}<a href="/app/viz.html?deal={}{}">All Pass</a></span>&nbsp;&nbsp;&nbsp;'.format(board_no_index,deal_id, board_no_ref)
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
    try:
        db = shelve.open(DB_NAME)
        deal = db[deal_id]
        db.close()

        return json.dumps(deal)
    except KeyError:
        # Handle the case when the specified deal_id does not exist
        error_message = {"error": f"Deal with ID {deal_id} not found"}
        return json.dumps(error_message), 404  # Return a 404 Not Found status

@app.route('/api/delete/deal/<deal_id>')
def delete_deal(deal_id):
    db = shelve.open(DB_NAME)
    db.pop(deal_id)
    db.close()
    redirect('/home')

@app.route('/api/save/deal', method='POST')
def save_deal():
    data_dict = request.json  # Get JSON data from the request body
    if data_dict:
        db = shelve.open(DB_NAME)
        db[uuid.uuid4().hex] = data_dict
        db.close()
        response.status = 200  # HTTP status code: 200 OK
        response.headers['Content-Type'] = 'application/json'  # Set response content type
        return json.dumps({'message': 'Deal saved successfully'})
    else:
        response.status = 400  # HTTP status code: 400 Bad Request
        response.headers['Content-Type'] = 'application/json'  # Set response content type
        return json.dumps({'error': 'Invalid data received'})

host = args.host
print(f'http://{host}:{port}/home')

run(app, host=host, port=port, server='gevent', log=None)
