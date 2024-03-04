import os
import shelve
import argparse
import json

parser = argparse.ArgumentParser(description="Appserver")
parser.add_argument("--db", default="../gamedb", help="Db for appserver")

args = parser.parse_args()

DB_NAME = os.getcwd() + "/" + args.db
print("Reading deals from: "+DB_NAME)

if __name__ == '__main__':

    deals = []
    with shelve.open(DB_NAME) as db:
        deal_items = sorted(list(db.items()), key=lambda x: x[1]['timestamp'])
        with open('log.js', 'w', encoding='utf-8') as file:  # Open the output file with UTF-8 encoding
            file.write("var data = ")
            for deal_id, deal in deal_items:
                deals.append(deal)
            file.write(json.dumps(deals))
            print("File log.js generated")
