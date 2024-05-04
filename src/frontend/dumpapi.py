import os
import shelve
import argparse
import json

parser = argparse.ArgumentParser(description="Appserver")
parser.add_argument("--db", default="../gameapibiddb", help="Db Name")

args = parser.parse_args()

DB_NAME = os.getcwd() + "/" + args.db
print("Reading deals from: "+DB_NAME)

if __name__ == '__main__':

    deals = []
    with shelve.open(DB_NAME) as db:
        deal_items = list(db.items())
        with open('api.json', 'w', encoding='utf-8') as file:  # Open the output file with UTF-8 encoding
            for deal_id, deal in deal_items:
                deals.append(deal)
            file.write(json.dumps(deals))
            print("File api.json generated")
        