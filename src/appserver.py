from gevent import monkey
monkey.patch_all()

from bottle import Bottle, run, static_file, redirect

import shelve
import json

app = Bottle()

DB_NAME = 'gamedb'

@app.route('/home')
def home():
    html = '<h1><a href="/app/bridge.html">Play Now</a></h1>\n'

    html += '<ul>\n'

    with shelve.open(DB_NAME) as db:
        deal_items = sorted(list(db.items()), key=lambda x: x[1]['timestamp'], reverse=True)

        for deal_id, deal in deal_items:
            html += '<li><span><a href="/app/viz.html?deal={}">{} {}</a></span>\n'.format(deal_id, deal['contract'], len(list(filter(lambda x: x % 2 == 1, deal['trick_winners']))))
            html += '<span><a href="/api/delete/deal/{}">delete</a></span></li>'.format(deal_id)

    html += '</ul>'
    return html

@app.route('/app/<filename>')
def frontend(filename):
    if '?' in filename:
        filename = filename[:filename.index('?')]
    return static_file(filename, root='./frontend')

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


run(app, host='0.0.0.0', port=8080, server='gevent')
