import itertools
import sys
sys.path.append('../../../src')

from ddsolver import ddsolver

import shelve
import os
import argparse
import random

os.getcwd()

parser = argparse.ArgumentParser(description="DDOLAR calculator")
parser.add_argument("--db", default="gamedb", help="Database with boards played by BEN")
parser.add_argument("--ns", default="True", help="Calculate for NS")

args = parser.parse_args()

# Convert the string to a boolean
NS = args.ns.lower() == "true"

DB_NAME = os.getcwd() + "/" + args.db

def encode_card(card_str):
    suit_i = 'SHDC'.index(card_str[0])
    card_i = 'AKQJT98765432'.index(card_str[1])
    return suit_i*13 + card_i

def decode_card(card52):
    suit_i = card52 // 13
    card_i = card52 % 13
    return 'SHDC'[suit_i] + 'AKQJT98765432'[card_i]

def get_strain_i(contract):
    return 'NSHDC'.index(contract[1])

def get_decl_i(decl):
    return 'WNES'.index(decl)

def play_data_iterator(fin):
    lines = []
    for i, line in enumerate(fin):
        line = line.strip()
        if i % 4 == 0 and i > 0:
            yield (lines[0], lines[1], lines[2], lines[3])
            lines = []

        lines.append(line)

    yield (lines[0], lines[1], lines[2], lines[3])

def validate_lead(deal_str, outcome_str, play_str):
    contract = outcome_str.split(' ')[2]
    contract_parts = contract.split('.')
    decl_i = get_decl_i(contract_parts[2])
    leader_i = (decl_i + 1) % 4
    
    lead = play_str[0:2]
    #print(lead)
    dd = ddsolver.DDSolver(dds_mode=2)
    strain_i = get_strain_i(contract_parts[0])
    #print(strain_i)
    hands_pbn = [deal_str]
    dd_solved = dd.solve(strain_i, leader_i, [], hands_pbn)
    #print(deal_str)
    max_key = max(dd_solved, key=lambda k: dd_solved[k])
    max_value = dd_solved[max_key]
    # Get the first value in the dictionary
    first_value = next(iter(dd_solved.values()))
    # Check if all values are the same
    if all(value == first_value for value in dd_solved.values()):
        return 0
    #for key, value in dd_solved.items():
    #    print(f"{decode_card(key)}: {value}")
    if (dd_solved[encode_card(lead)] == max_value):
        return 1
    else: 
        return -1

def validate_lead_gamedb(deal_str, outcome_str, play_str):
    decl_i = get_decl_i(outcome_str[-1])
    leader_i = (decl_i + 1) % 4
    
    lead = play_str
    #print(lead)
    dd = ddsolver.DDSolver(dds_mode=2)
    strain_i = get_strain_i(outcome_str)
    #print(strain_i)
    hands_pbn = ["N:" + deal_str]
    dd_solved = dd.solve(strain_i, leader_i, [], hands_pbn)
    #print(deal_str)
    max_key = max(dd_solved, key=lambda k: dd_solved[k])
    max_value = dd_solved[max_key]
    # Get the first value in the dictionary
    first_value = next(iter(dd_solved.values()))

    # Check if all values are the same
    if all(value == first_value for value in dd_solved.values()):
        return 0
    #for key, value in dd_solved.items():
    #    print(f"{decode_card(key)}: {value}")

    random_key = random.choice(list(dd_solved.keys()))
    random_value = dd_solved[random_key]
    if lead == "??":
        if (random_value == max_value):
            return 1
        else: 
            return -1
    else:
        if (dd_solved[encode_card(lead)] == max_value):
            return 1
        else: 
            return -1


def calculate_random_lead():
    ddsplus = 0
    ddsminus = 0
    ddnplus = 0
    ddnminus = 0
    ddn = 0
    dds = 0
    with shelve.open(DB_NAME) as db:
        deal_items = sorted(list(db.items()), key=lambda x: x[1]['timestamp'], reverse=True)
        for deal_id, deal in deal_items:
            #print(deal["hands"])
            deal_str = deal["hands"]
            outcome_str = deal['contract']
            #print(outcome_str)
            if outcome_str is None:
                continue
            play_str = "??"
            #print(play_str)
            dd = validate_lead_gamedb(deal_str, outcome_str, play_str)
            if dd == 1:
                if (get_strain_i(outcome_str) == 0):
                    ddnplus += 1
                else:                   
                    ddsplus += 1
            if dd == -1:
                if (get_strain_i(outcome_str) == 0):
                    ddnminus += 1
                else:
                    ddsminus += 1
            if dd == 0:
                if (get_strain_i(outcome_str) == 0):
                    ddn += 1
                else:
                    dds += 1
        

        print(f"DDOLAR  (suit) {ddsplus + ddsminus + dds:>4} = {((ddsplus + dds)*100/ (ddsplus + ddsminus + dds)):.2f}%")
        print(f"DDOLAR  (NT)   {ddnplus + ddnminus + ddn:>4} = {((ddnplus + ddn)*100/ (ddnplus + ddnminus + ddn)):.2f}%")
        print(f"ADDOLAR (suit) {ddsplus + ddsminus:>4} = {((ddsplus * 100)/ (ddsplus + ddsminus)):.2f}%")
        print(f"ADDOLAR (NT)   {ddnplus + ddnminus:>4} = {((ddnplus * 100)/(ddnplus + ddnminus)):.2f}%")

def calculate_lead_BW5C():
    ddsplus = 0
    ddsminus = 0
    ddnplus = 0
    ddnminus = 0
    ddn = 0
    dds = 0
    for i, (deal_str, outcome_str, bidding, play_str) in enumerate(play_data_iterator(itertools.chain(
        open('../scripts/training/opening lead/data/BW5C_N.txt'),
        #open('../scripts/training/opening lead/data/BW5C_S.txt'),
        #open('../scripts/training/opening lead/data/JOS_N.txt'),
        #open('../scripts/training/opening lead/data/JOS_S.txt')
        ))):
        # if (i+1) % 500 == 0:
        #     print(f"DDOLAR ({i+1}) = ",ddnplus / (ddnplus + ddnminus))
        dd = validate_lead(deal_str, outcome_str, play_str)
        if dd == 1:
            ddnplus += 1
        if dd == -1:
            ddnminus += 1
        if dd == 0:
            ddn += 1

    for i, (deal_str, outcome_str, bidding, play_str) in enumerate(play_data_iterator(itertools.chain(
        #open('../scripts/training/opening lead/data/BW5C_N.txt'),
        open('../scripts/training/opening lead/data/BW5C_S.txt'),
        #open('../scripts/training/opening lead/data/JOS_N.txt'),
        #open('../scripts/training/opening lead/data/JOS_S.txt')
        ))):
        # if (i+1) % 500 == 0:
        #     print(f"DDOLAR ({i+1}) = ",ddsplus / (ddsplus + ddsminus))
        dd = validate_lead(deal_str, outcome_str, play_str)
        if dd == 1:
            ddsplus += 1
        if dd == -1:
            ddsminus += 1
        if dd == 0:
            dds += 1

    print(f"DDOLAR  (suit) {ddsplus + ddsminus + dds:>4} = {((ddsplus + dds)*100/ (ddsplus + ddsminus + dds)):.2f}%")
    print(f"DDOLAR  (NT)   {ddnplus + ddnminus + ddn:>4} = {((ddnplus + ddn)*100/ (ddnplus + ddnminus + ddn)):.2f}%")
    print(f"ADDOLAR (suit) {ddsplus + ddsminus:>4} = {((ddsplus * 100)/ (ddsplus + ddsminus)):.2f}%")
    print(f"ADDOLAR (NT)   {ddnplus + ddnminus:>4} = {((ddnplus * 100)/(ddnplus + ddnminus)):.2f}%")

def calculate_lead_JOS():
    ddsplus = 0
    ddsminus = 0
    ddnplus = 0
    ddnminus = 0
    ddn = 0
    dds = 0
    save_deal_N = []
    save_deal_S = []
    for i, (deal_str, outcome_str, bidding, play_str) in enumerate(play_data_iterator(itertools.chain(
        #open('../scripts/training/opening lead/data/BW5C_N.txt'),
        #open('../scripts/training/opening lead/data/BW5C_S.txt'),
        open('../scripts/training/opening lead/data/JOS_N.txt'),
        #open('../scripts/training/opening lead/data/JOS_S.txt')
        ))):
        if (i+1) % 500 == 0:
            print(f"DDOLAR ({i+1}) = ",ddnplus / (ddnplus + ddnminus))
        dd = validate_lead(deal_str, outcome_str, play_str)
        if dd == 1:
            ddnplus += 1
            save_deal_N.append((deal_str, outcome_str, bidding, play_str))
        if dd == -1:
            ddnminus += 1
        if dd == 0:
            ddn += 1

    # Step 3: Write values to a new file when the program ends
    with open('SuitLead.txt', 'w') as output_file:
        for deal_str, outcome_str, bidding, play_str in save_deal_S:
            output_file.write(f"{deal_str}\n")
            output_file.write(f"{outcome_str}\n")
            output_file.write(f"{bidding}\n")
            output_file.write(f"{play_str}\n")

    print(f"DDOLAR  (NT)   {ddnplus + ddnminus + ddn:>4} = {((ddnplus + ddn)*100/ (ddnplus + ddnminus + ddn)):.2f}%")
    print(f"ADDOLAR (NT)   {ddnplus + ddnminus:>4} = {((ddnplus * 100)/(ddnplus + ddnminus)):.2f}%")

    for i, (deal_str, outcome_str, bidding, play_str) in enumerate(play_data_iterator(itertools.chain(
        #open('../scripts/training/opening lead/data/BW5C_N.txt'),
        #open('../scripts/training/opening lead/data/BW5C_S.txt'),
        #open('../scripts/training/opening lead/data/JOS_N.txt'),
        open('../scripts/training/opening lead/data/JOS_S.txt')
        ))):
        if (i+1) % 500 == 0:
             print(f"DDOLAR ({i+1}) = ",ddsplus / (ddsplus + ddsminus))
             #break
        dd = validate_lead(deal_str, outcome_str, play_str)
        if dd == 1:
            ddsplus += 1
            save_deal_S.append((deal_str, outcome_str, bidding, play_str))
        if dd == -1:
            ddsminus += 1
        if dd == 0:
            dds += 1

    with open('NT_Lead.txt', 'w') as output_file:
        for deal_str, outcome_str, bidding, play_str in save_deal_N:
            output_file.write(f"{deal_str}\n")
            output_file.write(f"{outcome_str}\n")
            output_file.write(f"{bidding}\n")
            output_file.write(f"{play_str}\n")

    print(f"DDOLAR  (suit) {ddsplus + ddsminus + dds:>4} = {((ddsplus + dds)*100/ (ddsplus + ddsminus + dds)):.2f}%")
    print(f"ADDOLAR (suit) {ddsplus + ddsminus:>4} = {((ddsplus * 100)/ (ddsplus + ddsminus)):.2f}%")

def calculate_lead_BEN():
    ddsplus = 0
    ddsminus = 0
    ddnplus = 0
    ddnminus = 0
    ddn = 0
    dds = 0
    i = 0
    j = 0
    with shelve.open(DB_NAME) as db:
        deal_items = sorted(list(db.items()), key=lambda x: x[1]['timestamp'], reverse=True)
        for deal_id, deal in deal_items:
            j += 1
            #print(deal["hands"])
            deal_str = deal["hands"]
            outcome_str = deal['contract']
            #print(outcome_str)
            if outcome_str is None:
                continue
            decl_i = get_decl_i(outcome_str[-1])
            if NS:
                if decl_i == 1 or decl_i == 3:
                    # We are declaring so skip the deal
                    continue
            else:
                if decl_i == 0 or decl_i == 2:
                    # We are declaring so skip the deal
                    continue

            play_str = deal["play"][0]['card']         

            #print(play_str, deal["play"][0]['candidates'][0]['card'], deal["play"][0]['candidates'][0]['insta_score'])
            dd = validate_lead_gamedb(deal_str, outcome_str, play_str)
            #if dd == -1 and deal["play"][0]['candidates'][0]['insta_score'] < 0.2:
            #    print("Wrong card", deal["play"][0]['candidates'][0]['insta_score'])
            #if dd == 1 and deal["play"][0]['candidates'][0]['insta_score'] < 0.2:
            #    print("Right card", deal["play"][0]['candidates'][0]['insta_score'])
            if dd == 1:
                if (get_strain_i(outcome_str) == 0):
                    ddnplus += 1
                else:                   
                    ddsplus += 1
            if dd == -1:
                if (get_strain_i(outcome_str) == 0):
                    ddnminus += 1
                else:
                    ddsminus += 1
            if dd == 0:
                if (get_strain_i(outcome_str) == 0):
                    ddn += 1
                else:
                    dds += 1
            i = i + 1
            if i > 1000:
                break
        
        print(f"Calculated on {i} of {j} deals.")
        print(f"DDOLAR  (suit) {ddsplus + ddsminus + dds:>4} = {((ddsplus + dds)*100/ (ddsplus + ddsminus + dds)):.2f}%")
        print(f"DDOLAR  (NT)   {ddnplus + ddnminus + ddn:>4} = {((ddnplus + ddn)*100/ (ddnplus + ddnminus + ddn)):.2f}%")
        print(f"ADDOLAR (suit) {ddsplus + ddsminus:>4} = {((ddsplus * 100)/ (ddsplus + ddsminus)):.2f}%")
        print(f"ADDOLAR (NT)   {ddnplus + ddnminus:>4} = {((ddnplus * 100)/(ddnplus + ddnminus)):.2f}%")
        print()
        print(f"DDOLAR         {ddsplus + ddsminus + dds + ddnplus + ddnminus + ddn:>4} = {((ddsplus + dds+ddnplus + ddn)*100/ (ddsplus + ddsminus + dds+ddnplus + ddnminus + ddn)):.2f}%")
        print(f"ADDOLAR        {ddsplus + ddsminus + ddnplus + ddnminus:>4} = {(((ddsplus+ddnplus) * 100)/ (ddsplus + ddsminus+ddnplus + ddnminus)):.2f}%")

if __name__ == '__main__':

    #calculate_random_lead()

    #calculate_lead_BW5C()

    #calculate_lead_JOS()

    calculate_lead_BEN()

